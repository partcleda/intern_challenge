#include "placement/sqlite_utils.hpp"

#include "placement/types.h"

#include <sqlite3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace placement {

std::string sqliteError(sqlite3* db) {
    const char* message = db == nullptr ? nullptr : sqlite3_errmsg(db);
    return message == nullptr ? "unknown sqlite error" : std::string(message);
}

[[noreturn]] void throwSqliteError(sqlite3* db, std::string_view context) {
    throw std::runtime_error(std::string(context) + ": " + sqliteError(db));
}

void checkSqliteResult(
    sqlite3* db,
    int result_code,
    std::string_view context) {
    if (result_code != SQLITE_OK) {
        throwSqliteError(db, context);
    }
}

void executeSql(sqlite3* db, std::string_view sql, std::string_view context) {
    char* error_message = nullptr;
    const std::string sql_text(sql);
    const int result_code =
        sqlite3_exec(db, sql_text.c_str(), nullptr, nullptr, &error_message);
    if (result_code != SQLITE_OK) {
        const std::string message =
            error_message == nullptr ? sqliteError(db) : std::string(error_message);
        sqlite3_free(error_message);
        throw std::runtime_error(std::string(context) + ": " + message);
    }
}

SqliteConnection::SqliteConnection(const std::filesystem::path& db_path) {
    sqlite3* opened_db = nullptr;
    const int result_code = sqlite3_open(db_path.string().c_str(), &opened_db);
    db_ = opened_db;
    if (result_code != SQLITE_OK) {
        const std::string message = sqliteError(db_);
        if (db_ != nullptr) {
            sqlite3_close(db_);
            db_ = nullptr;
        }
        throw std::runtime_error(
            "Unable to open SQLite database " + db_path.string() + ": " +
            message);
    }
    executeSql(db_, "PRAGMA foreign_keys = ON", "Enable SQLite foreign keys");
    checkSqliteResult(
        db_,
        sqlite3_busy_timeout(db_, 30000),
        "Set SQLite busy timeout");
    executeSql(
        db_,
        "PRAGMA journal_mode = WAL",
        "Enable SQLite WAL journal mode");
}

SqliteConnection::~SqliteConnection() {
    if (db_ != nullptr) {
        sqlite3_close(db_);
    }
}

sqlite3* SqliteConnection::get() const {
    return db_;
}

SqliteStatement::SqliteStatement(sqlite3* db, std::string_view sql) : db_(db) {
    const std::string sql_text(sql);
    const int result_code =
        sqlite3_prepare_v2(db_, sql_text.c_str(), -1, &stmt_, nullptr);
    checkSqliteResult(db_, result_code, "Prepare SQLite statement");
}

SqliteStatement::~SqliteStatement() {
    if (stmt_ != nullptr) {
        sqlite3_finalize(stmt_);
    }
}

sqlite3_stmt* SqliteStatement::get() const {
    return stmt_;
}

bool SqliteStatement::stepRow() {
    const int result_code = sqlite3_step(stmt_);
    if (result_code == SQLITE_ROW) {
        return true;
    }
    if (result_code == SQLITE_DONE) {
        return false;
    }
    throwSqliteError(db_, "Step SQLite row statement");
}

void SqliteStatement::stepDone() {
    const int result_code = sqlite3_step(stmt_);
    if (result_code != SQLITE_DONE) {
        throwSqliteError(db_, "Step SQLite write statement");
    }
}

void SqliteStatement::reset() {
    checkSqliteResult(db_, sqlite3_reset(stmt_), "Reset SQLite statement");
    checkSqliteResult(
        db_,
        sqlite3_clear_bindings(stmt_),
        "Clear SQLite statement bindings");
}

void bindNull(sqlite3_stmt* stmt, int index) {
    checkSqliteResult(
        sqlite3_db_handle(stmt),
        sqlite3_bind_null(stmt, index),
        "Bind SQLite null");
}

void bindText(sqlite3_stmt* stmt, int index, std::string_view value) {
    const std::string value_text(value);
    checkSqliteResult(
        sqlite3_db_handle(stmt),
        sqlite3_bind_text(
            stmt,
            index,
            value_text.c_str(),
            -1,
            SQLITE_TRANSIENT),
        "Bind SQLite text");
}

void bindInt64(sqlite3_stmt* stmt, int index, int64_t value) {
    checkSqliteResult(
        sqlite3_db_handle(stmt),
        sqlite3_bind_int64(stmt, index, static_cast<sqlite3_int64>(value)),
        "Bind SQLite integer");
}

void bindBool(sqlite3_stmt* stmt, int index, bool value) {
    bindInt64(stmt, index, value ? 1 : 0);
}

void bindDouble(sqlite3_stmt* stmt, int index, double value) {
    if (!std::isfinite(value)) {
        bindNull(stmt, index);
        return;
    }
    checkSqliteResult(
        sqlite3_db_handle(stmt),
        sqlite3_bind_double(stmt, index, value),
        "Bind SQLite double");
}

void bindOptionalInt64(
    sqlite3_stmt* stmt,
    int index,
    const std::optional<int64_t>& value) {
    if (value.has_value()) {
        bindInt64(stmt, index, *value);
    } else {
        bindNull(stmt, index);
    }
}

void ensureColumns(
    sqlite3* db,
    std::string_view table_name,
    const std::vector<std::pair<std::string, std::string>>& columns) {
    std::vector<std::string> existing_columns;
    SqliteStatement statement(
        db,
        "PRAGMA table_info(" + std::string(table_name) + ")");
    while (statement.stepRow()) {
        const unsigned char* column_text =
            sqlite3_column_text(statement.get(), 1);
        if (column_text != nullptr) {
            existing_columns.emplace_back(
                reinterpret_cast<const char*>(column_text));
        }
    }

    for (const auto& [column_name, column_type] : columns) {
        if (std::find(
                existing_columns.begin(),
                existing_columns.end(),
                column_name) == existing_columns.end()) {
            executeSql(
                db,
                "ALTER TABLE " + std::string(table_name) + " ADD COLUMN " +
                    column_name + " " + column_type,
                "Add SQLite schema column");
        }
    }
}

namespace {

std::tm localTime(std::time_t time) {
    std::tm local_time{};
#if defined(_WIN32)
    localtime_s(&local_time, &time);
#else
    localtime_r(&time, &local_time);
#endif
    return local_time;
}

std::string compactTimestamp(std::chrono::system_clock::time_point timestamp) {
    const std::time_t now_time =
        std::chrono::system_clock::to_time_t(timestamp);
    const std::tm local_time = localTime(now_time);
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
                            timestamp.time_since_epoch()) %
                        std::chrono::seconds(1);

    std::ostringstream output;
    output << std::put_time(&local_time, "%Y%m%d_%H%M%S_")
           << std::setw(6) << std::setfill('0') << micros.count();
    return output.str();
}

std::string isoTimestampSeconds(
    std::chrono::system_clock::time_point timestamp) {
    const std::time_t now_time =
        std::chrono::system_clock::to_time_t(timestamp);
    const std::tm local_time = localTime(now_time);

    std::ostringstream output;
    output << std::put_time(&local_time, "%Y-%m-%dT%H:%M:%S");
    return output.str();
}

std::filesystem::path repoRootPath() {
#if defined(PLACEMENT_REPO_ROOT)
    return std::filesystem::path(PLACEMENT_REPO_ROOT).lexically_normal();
#else
    return std::filesystem::path(__FILE__).parent_path().parent_path();
#endif
}

std::filesystem::path lossTrackingDbDir() {
    return repoRootPath() / "loss_tracking";
}

std::mutex& lossTrackingMutex() {
    static std::mutex mutex;
    return mutex;
}

void bindLossDouble(
    sqlite3_stmt* stmt,
    int index,
    const std::vector<double>& values,
    std::size_t epoch) {
    if (epoch < values.size()) {
        bindDouble(stmt, index, values[epoch]);
    } else {
        bindNull(stmt, index);
    }
}

void bindLossInt(
    sqlite3_stmt* stmt,
    int index,
    const std::vector<int>& values,
    std::size_t epoch) {
    if (epoch < values.size()) {
        bindInt64(stmt, index, values[epoch]);
    } else {
        bindNull(stmt, index);
    }
}

void initializeLossTrackingSchema(sqlite3* db) {
    executeSql(
        db,
        R"sql(
        CREATE TABLE IF NOT EXISTS test_cases (
            test_id INTEGER PRIMARY KEY,
            num_macros INTEGER,
            num_std_cells INTEGER,
            seed INTEGER,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            test_id INTEGER REFERENCES test_cases(test_id) ON DELETE SET NULL,
            runner TEXT,
            run_label TEXT,
            run_started_at TEXT,
            saved_at TEXT NOT NULL,
            seed INTEGER,
            num_macros INTEGER,
            num_std_cells INTEGER,
            num_epochs INTEGER,
            lr REAL,
            lambda_wirelength REAL,
            lambda_overlap REAL,
            log_interval INTEGER,
            verbose INTEGER,
            total_cells INTEGER,
            total_pins INTEGER,
            total_edges INTEGER
        );

        CREATE TABLE IF NOT EXISTS loss_history (
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            epoch INTEGER NOT NULL,
            total_loss REAL,
            wirelength_loss REAL,
            overlap_loss REAL,
            learning_rate REAL,
            overlap_count INTEGER,
            total_overlap_area REAL,
            max_overlap_area REAL,
            PRIMARY KEY (run_id, epoch)
        );
        )sql",
        "Initialize loss-tracking schema");

    ensureColumns(
        db,
        "runs",
        {
            {"seed", "INTEGER"},
            {"num_macros", "INTEGER"},
            {"num_std_cells", "INTEGER"},
        });
    ensureColumns(
        db,
        "loss_history",
        {
            {"learning_rate", "REAL"},
            {"overlap_count", "INTEGER"},
            {"total_overlap_area", "REAL"},
            {"max_overlap_area", "REAL"},
        });
}

std::size_t lossHistoryRowCount(const LossHistory& history) {
    return std::max(
        {history.total_loss.size(),
         history.wirelength_loss.size(),
         history.overlap_loss.size(),
         history.learning_rate.size(),
         history.overlap_count.size(),
         history.total_overlap_area.size(),
         history.max_overlap_area.size()});
}

}  // namespace

std::filesystem::path createLossTrackingDb() {
    const std::filesystem::path db_dir = lossTrackingDbDir();
    std::filesystem::create_directories(db_dir);
    const auto now = std::chrono::system_clock::now();
    const std::filesystem::path db_path =
        db_dir / ("loss_tracking_" + compactTimestamp(now) + ".sqlite3");

    SqliteConnection connection(db_path);
    initializeLossTrackingSchema(connection.get());
    return db_path;
}

std::filesystem::path saveLossHistorySqlite(
    const LossHistory& history,
    const std::filesystem::path& db_path,
    const LossHistoryRunMetadata& metadata) {
    std::lock_guard<std::mutex> lock(lossTrackingMutex());
    std::filesystem::create_directories(db_path.parent_path());

    const auto now = std::chrono::system_clock::now();
    const std::string saved_at = isoTimestampSeconds(now);
    const std::string run_id = compactTimestamp(now);
    const std::string run_started_at =
        metadata.run_started_at.empty() ? saved_at : metadata.run_started_at;

    SqliteConnection connection(db_path);
    sqlite3* db = connection.get();
    initializeLossTrackingSchema(db);

    executeSql(db, "BEGIN IMMEDIATE", "Begin loss-history transaction");
    try {
        if (metadata.test_id.has_value()) {
            SqliteStatement test_case_statement(
                db,
                R"sql(
                INSERT INTO test_cases (
                    test_id,
                    num_macros,
                    num_std_cells,
                    seed,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(test_id) DO UPDATE SET
                    num_macros = excluded.num_macros,
                    num_std_cells = excluded.num_std_cells,
                    seed = excluded.seed,
                    updated_at = excluded.updated_at
                )sql");
            bindInt64(test_case_statement.get(), 1, *metadata.test_id);
            bindInt64(test_case_statement.get(), 2, metadata.num_macros);
            bindInt64(test_case_statement.get(), 3, metadata.num_std_cells);
            bindInt64(test_case_statement.get(), 4, metadata.seed);
            bindText(test_case_statement.get(), 5, saved_at);
            test_case_statement.stepDone();
        }

        SqliteStatement run_statement(
            db,
            R"sql(
            INSERT OR REPLACE INTO runs (
                run_id,
                test_id,
                runner,
                run_label,
                run_started_at,
                saved_at,
                seed,
                num_macros,
                num_std_cells,
                num_epochs,
                lr,
                lambda_wirelength,
                lambda_overlap,
                log_interval,
                verbose,
                total_cells,
                total_pins,
                total_edges
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            )sql");
        bindText(run_statement.get(), 1, run_id);
        bindOptionalInt64(
            run_statement.get(),
            2,
            metadata.test_id.has_value()
                ? std::optional<int64_t>(*metadata.test_id)
                : std::nullopt);
        bindText(run_statement.get(), 3, metadata.runner);
        bindText(run_statement.get(), 4, metadata.run_label);
        bindText(run_statement.get(), 5, run_started_at);
        bindText(run_statement.get(), 6, saved_at);
        bindInt64(run_statement.get(), 7, metadata.seed);
        bindInt64(run_statement.get(), 8, metadata.num_macros);
        bindInt64(run_statement.get(), 9, metadata.num_std_cells);
        bindInt64(run_statement.get(), 10, metadata.num_epochs);
        bindDouble(run_statement.get(), 11, metadata.lr);
        bindDouble(run_statement.get(), 12, metadata.lambda_wirelength);
        bindDouble(run_statement.get(), 13, metadata.lambda_overlap);
        bindInt64(run_statement.get(), 14, metadata.log_interval);
        bindBool(run_statement.get(), 15, metadata.verbose);
        bindInt64(run_statement.get(), 16, metadata.total_cells);
        bindInt64(run_statement.get(), 17, metadata.total_pins);
        bindInt64(run_statement.get(), 18, metadata.total_edges);
        run_statement.stepDone();

        SqliteStatement delete_statement(
            db,
            "DELETE FROM loss_history WHERE run_id = ?");
        bindText(delete_statement.get(), 1, run_id);
        delete_statement.stepDone();

        SqliteStatement history_statement(
            db,
            R"sql(
            INSERT INTO loss_history (
                run_id,
                epoch,
                total_loss,
                wirelength_loss,
                overlap_loss,
                learning_rate,
                overlap_count,
                total_overlap_area,
                max_overlap_area
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            )sql");
        const std::size_t row_count = lossHistoryRowCount(history);
        for (std::size_t epoch = 0; epoch < row_count; ++epoch) {
            history_statement.reset();
            bindText(history_statement.get(), 1, run_id);
            bindInt64(history_statement.get(), 2, static_cast<int64_t>(epoch));
            bindLossDouble(history_statement.get(), 3, history.total_loss, epoch);
            bindLossDouble(
                history_statement.get(),
                4,
                history.wirelength_loss,
                epoch);
            bindLossDouble(history_statement.get(), 5, history.overlap_loss, epoch);
            bindLossDouble(history_statement.get(), 6, history.learning_rate, epoch);
            bindLossInt(history_statement.get(), 7, history.overlap_count, epoch);
            bindLossDouble(
                history_statement.get(),
                8,
                history.total_overlap_area,
                epoch);
            bindLossDouble(
                history_statement.get(),
                9,
                history.max_overlap_area,
                epoch);
            history_statement.stepDone();
        }

        executeSql(db, "COMMIT", "Commit loss-history transaction");
    } catch (...) {
        executeSql(db, "ROLLBACK", "Rollback loss-history transaction");
        throw;
    }

    return db_path;
}

}  // namespace placement
