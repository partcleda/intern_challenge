#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

struct sqlite3;
struct sqlite3_stmt;

namespace placement {

struct LossHistory;

struct LossHistoryRunMetadata {
    std::optional<int> test_id;
    std::string runner;
    std::string run_label = "train_placement";
    std::string run_started_at;
    int seed = 0;
    int num_macros = 0;
    int num_std_cells = 0;
    int num_epochs = 0;
    double lr = 0.0;
    double lambda_wirelength = 0.0;
    double lambda_overlap = 0.0;
    int log_interval = 0;
    bool verbose = false;
    int64_t total_cells = 0;
    int64_t total_pins = 0;
    int64_t total_edges = 0;
};

std::string sqliteError(sqlite3* db);

[[noreturn]] void throwSqliteError(sqlite3* db, std::string_view context);

void checkSqliteResult(
    sqlite3* db,
    int result_code,
    std::string_view context);

void executeSql(sqlite3* db, std::string_view sql, std::string_view context);

class SqliteConnection {
public:
    explicit SqliteConnection(const std::filesystem::path& db_path);

    SqliteConnection(const SqliteConnection&) = delete;
    SqliteConnection& operator=(const SqliteConnection&) = delete;

    ~SqliteConnection();

    sqlite3* get() const;

private:
    sqlite3* db_ = nullptr;
};

class SqliteStatement {
public:
    SqliteStatement(sqlite3* db, std::string_view sql);

    SqliteStatement(const SqliteStatement&) = delete;
    SqliteStatement& operator=(const SqliteStatement&) = delete;

    ~SqliteStatement();

    sqlite3_stmt* get() const;
    bool stepRow();
    void stepDone();
    void reset();

private:
    sqlite3* db_ = nullptr;
    sqlite3_stmt* stmt_ = nullptr;
};

void bindNull(sqlite3_stmt* stmt, int index);
void bindText(sqlite3_stmt* stmt, int index, std::string_view value);
void bindInt64(sqlite3_stmt* stmt, int index, int64_t value);
void bindBool(sqlite3_stmt* stmt, int index, bool value);
void bindDouble(sqlite3_stmt* stmt, int index, double value);
void bindOptionalInt64(
    sqlite3_stmt* stmt,
    int index,
    const std::optional<int64_t>& value);

void ensureColumns(
    sqlite3* db,
    std::string_view table_name,
    const std::vector<std::pair<std::string, std::string>>& columns);

std::filesystem::path createLossTrackingDb();

std::filesystem::path saveLossHistorySqlite(
    const LossHistory& history,
    const std::filesystem::path& db_path,
    const LossHistoryRunMetadata& metadata);

}  // namespace placement
