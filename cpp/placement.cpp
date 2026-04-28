#include "placement/benchmark.h"
#include "placement/generation.h"
#include "placement/metrics.h"
#include "placement/sqlite_utils.hpp"
#include "placement/training.h"
#include "placement/types.h"
#include "placement/visualization.h"

#include <CLI/CLI.hpp>
#include <torch/cuda.h>
#include <torch/mps.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using placement::LossHistoryRunMetadata;
using placement::createLossTrackingDb;
using placement::saveLossHistorySqlite;

struct CliOptions {
    bool run_benchmark = false;
    std::string device = "auto";
    int test_case_id = 0;
    int num_macros = 3;
    int num_std_cells = 10;
    int seed = 42;
    bool write_output_files = false;
    std::string output_dir = "..";
};

std::string deviceTypeName(c10::DeviceType device) {
    switch (device) {
        case c10::DeviceType::CPU:
            return "cpu";
        case c10::DeviceType::CUDA:
            return "cuda";
        case c10::DeviceType::MPS:
            return "mps";
        default:
            return "unknown";
    }
}

c10::DeviceType resolveDeviceType(const std::string& device) {
    if (device == "auto") {
        if (torch::cuda::is_available()) {
            return torch::kCUDA;
        }
        if (torch::mps::is_available()) {
            return torch::kMPS;
        }
        return torch::kCPU;
    }
    if (device == "cpu") {
        return torch::kCPU;
    }
    if (device == "cuda") {
        if (!torch::cuda::is_available()) {
            throw std::invalid_argument("CUDA device requested but unavailable");
        }
        return torch::kCUDA;
    }
    if (device == "mps") {
        if (!torch::mps::is_available()) {
            throw std::invalid_argument("MPS device requested but unavailable");
        }
        return torch::kMPS;
    }
    throw std::invalid_argument("Unsupported device: " + device);
}

void seedTorch(int seed) {
    torch::manual_seed(static_cast<uint64_t>(seed));
    if (torch::cuda::is_available()) {
        torch::cuda::manual_seed_all(static_cast<uint64_t>(seed));
    }
    if (torch::mps::is_available()) {
        torch::mps::manual_seed(static_cast<uint64_t>(seed));
    }
}

std::vector<placement::BenchmarkCase> allKnownBenchmarkCases() {
    std::vector<placement::BenchmarkCase> cases(
        placement::activeBenchmarkCases().begin(),
        placement::activeBenchmarkCases().end());
    cases.push_back({11, 10, 10000, 1011});
    cases.push_back({12, 10, 100000, 1012});
    return cases;
}

std::optional<placement::BenchmarkCase> findBenchmarkCase(int test_case_id) {
    for (const placement::BenchmarkCase& test_case : allKnownBenchmarkCases()) {
        if (test_case.test_id == test_case_id) {
            return test_case;
        }
    }
    return std::nullopt;
}

std::string formatDouble(double value) {
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
}

std::string boolText(bool value) {
    return value ? "true" : "false";
}

std::string csvEscape(std::string_view value) {
    if (value.find_first_of("\",\n\r") == std::string_view::npos) {
        return std::string(value);
    }

    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (char ch : value) {
        if (ch == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
}

std::string jsonEscape(std::string_view value) {
    std::ostringstream escaped;
    for (unsigned char ch : value) {
        switch (ch) {
            case '"':
                escaped << "\\\"";
                break;
            case '\\':
                escaped << "\\\\";
                break;
            case '\b':
                escaped << "\\b";
                break;
            case '\f':
                escaped << "\\f";
                break;
            case '\n':
                escaped << "\\n";
                break;
            case '\r':
                escaped << "\\r";
                break;
            case '\t':
                escaped << "\\t";
                break;
            default:
                if (ch < 0x20) {
                    escaped << "\\u" << std::hex << std::setw(4)
                            << std::setfill('0') << static_cast<int>(ch)
                            << std::dec << std::setfill(' ');
                } else {
                    escaped << static_cast<char>(ch);
                }
                break;
        }
    }
    return escaped.str();
}

std::string jsonString(std::string_view value) {
    std::string quoted = "\"";
    quoted += jsonEscape(value);
    quoted += "\"";
    return quoted;
}

std::string jsonDouble(double value) {
    if (!std::isfinite(value)) {
        return "null";
    }
    return formatDouble(value);
}

std::string jsonBool(bool value) {
    return boolText(value);
}

std::filesystem::path outputFilePath(
    const CliOptions& options,
    std::string_view file_name) {
    const std::filesystem::path output_dir =
        options.output_dir.empty() ? std::filesystem::path(".")
                                   : std::filesystem::path(options.output_dir);
    return output_dir / std::string(file_name);
}

void writeTextFile(
    const std::filesystem::path& file_path,
    const std::string& contents) {
    const std::filesystem::path parent_path = file_path.parent_path();
    if (!parent_path.empty()) {
        std::filesystem::create_directories(parent_path);
    }

    std::ofstream output(file_path, std::ios::out | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "Unable to open output file: " + file_path.string());
    }
    output << contents;
    if (!output) {
        throw std::runtime_error(
            "Unable to write output file: " + file_path.string());
    }
}

void appendCsvRow(
    std::ostringstream& output,
    const std::vector<std::string>& fields) {
    for (std::size_t index = 0; index < fields.size(); ++index) {
        if (index > 0) {
            output << ",";
        }
        output << csvEscape(fields[index]);
    }
    output << "\n";
}

void writeCsvFile(
    const std::filesystem::path& file_path,
    const std::vector<std::string>& header,
    const std::vector<std::vector<std::string>>& rows) {
    std::ostringstream output;
    appendCsvRow(output, header);
    for (const std::vector<std::string>& row : rows) {
        appendCsvRow(output, row);
    }
    writeTextFile(file_path, output.str());
}

void appendJsonField(
    std::ostringstream& output,
    int indent,
    std::string_view key,
    const std::string& value,
    bool trailing_comma) {
    output << std::string(indent, ' ') << jsonString(key) << ": " << value;
    if (trailing_comma) {
        output << ",";
    }
    output << "\n";
}

using JsonField = std::pair<std::string, std::string>;

void appendJsonObject(
    std::ostringstream& output,
    const std::vector<JsonField>& fields,
    int indent) {
    output << std::string(indent, ' ') << "{\n";
    for (std::size_t index = 0; index < fields.size(); ++index) {
        appendJsonField(
            output,
            indent + 2,
            fields[index].first,
            fields[index].second,
            index + 1 < fields.size());
    }
    output << std::string(indent, ' ') << "}";
}

std::vector<std::string> benchmarkResultHeader() {
    return {
        "test_id",
        "num_macros",
        "num_std_cells",
        "total_cells",
        "num_nets",
        "seed",
        "device",
        "elapsed_seconds",
        "num_cells_with_overlaps",
        "overlap_ratio",
        "normalized_wl",
        "passed",
        "stopped_early",
        "stop_reason",
        "best_epoch",
        "epochs_completed",
    };
}

std::vector<std::string> benchmarkResultRow(
    const placement::BenchmarkResult& result) {
    return {
        std::to_string(result.test_id),
        std::to_string(result.num_macros),
        std::to_string(result.num_std_cells),
        std::to_string(result.total_cells),
        std::to_string(result.num_nets),
        std::to_string(result.seed),
        deviceTypeName(result.device),
        formatDouble(result.elapsed_seconds),
        std::to_string(result.num_cells_with_overlaps),
        formatDouble(result.overlap_ratio),
        formatDouble(result.normalized_wl),
        boolText(result.passed),
        boolText(result.stopped_early),
        result.stop_reason,
        std::to_string(result.best_epoch),
        std::to_string(result.epochs_completed),
    };
}

std::vector<JsonField> benchmarkResultJsonFields(
    const placement::BenchmarkResult& result) {
    return {
        {"test_id", std::to_string(result.test_id)},
        {"num_macros", std::to_string(result.num_macros)},
        {"num_std_cells", std::to_string(result.num_std_cells)},
        {"total_cells", std::to_string(result.total_cells)},
        {"num_nets", std::to_string(result.num_nets)},
        {"seed", std::to_string(result.seed)},
        {"device", jsonString(deviceTypeName(result.device))},
        {"elapsed_seconds", jsonDouble(result.elapsed_seconds)},
        {"num_cells_with_overlaps",
         std::to_string(result.num_cells_with_overlaps)},
        {"overlap_ratio", jsonDouble(result.overlap_ratio)},
        {"normalized_wl", jsonDouble(result.normalized_wl)},
        {"passed", jsonBool(result.passed)},
        {"stopped_early", jsonBool(result.stopped_early)},
        {"stop_reason", jsonString(result.stop_reason)},
        {"best_epoch", std::to_string(result.best_epoch)},
        {"epochs_completed", std::to_string(result.epochs_completed)},
    };
}

std::vector<std::string> singlePlacementHeader() {
    return {
        "run_type",
        "test_case_id",
        "seed",
        "device",
        "num_macros",
        "num_std_cells",
        "total_cells",
        "num_nets",
        "initial_overlap_count",
        "initial_total_overlap_area",
        "initial_max_overlap_area",
        "initial_overlap_percentage",
        "initial_cells_with_overlap",
        "initial_has_zero_overlap",
        "final_overlap_count",
        "final_total_overlap_area",
        "final_max_overlap_area",
        "final_overlap_percentage",
        "final_cells_with_overlap",
        "final_has_zero_overlap",
        "normalized_overlap_ratio",
        "normalized_wl",
        "normalized_num_cells_with_overlaps",
        "normalized_total_cells",
        "normalized_num_nets",
        "passed",
        "stopped_early",
        "stop_reason",
        "best_epoch",
        "epochs_completed",
        "num_epochs",
        "early_stop_enabled",
        "early_stop_patience",
        "early_stop_min_delta",
        "early_stop_overlap_threshold",
        "early_stop_zero_overlap_patience",
        "visualization_path",
    };
}

std::vector<std::string> singlePlacementRow(
    const placement::BenchmarkCase& selected_case,
    const placement::TrainingConfig& config,
    const placement::OverlapMetrics& initial_metrics,
    const placement::OverlapMetrics& final_metrics,
    const placement::Metrics& normalized_metrics,
    const placement::TrainingResult& training_result,
    bool passed,
    const std::filesystem::path& visualization_path) {
    return {
        "single",
        std::to_string(selected_case.test_id),
        std::to_string(selected_case.seed),
        deviceTypeName(config.device),
        std::to_string(selected_case.num_macros),
        std::to_string(selected_case.num_std_cells),
        std::to_string(normalized_metrics.total_cells),
        std::to_string(normalized_metrics.num_nets),
        std::to_string(initial_metrics.overlap_count),
        formatDouble(initial_metrics.total_overlap_area),
        formatDouble(initial_metrics.max_overlap_area),
        formatDouble(initial_metrics.overlap_percentage),
        std::to_string(initial_metrics.cells_with_overlap),
        boolText(initial_metrics.has_zero_overlap),
        std::to_string(final_metrics.overlap_count),
        formatDouble(final_metrics.total_overlap_area),
        formatDouble(final_metrics.max_overlap_area),
        formatDouble(final_metrics.overlap_percentage),
        std::to_string(final_metrics.cells_with_overlap),
        boolText(final_metrics.has_zero_overlap),
        formatDouble(normalized_metrics.overlap_ratio),
        formatDouble(normalized_metrics.normalized_wl),
        std::to_string(normalized_metrics.num_cells_with_overlaps),
        std::to_string(normalized_metrics.total_cells),
        std::to_string(normalized_metrics.num_nets),
        boolText(passed),
        boolText(training_result.stopped_early),
        training_result.stop_reason,
        std::to_string(training_result.best_epoch),
        std::to_string(training_result.epochs_completed),
        std::to_string(config.num_epochs),
        boolText(config.early_stop_enabled),
        std::to_string(config.early_stop_patience),
        formatDouble(config.early_stop_min_delta),
        formatDouble(config.early_stop_overlap_threshold),
        std::to_string(config.early_stop_zero_overlap_patience),
        visualization_path.string(),
    };
}

std::vector<JsonField> singlePlacementJsonFields(
    const placement::BenchmarkCase& selected_case,
    const placement::TrainingConfig& config,
    const placement::OverlapMetrics& initial_metrics,
    const placement::OverlapMetrics& final_metrics,
    const placement::Metrics& normalized_metrics,
    const placement::TrainingResult& training_result,
    bool passed,
    const std::filesystem::path& visualization_path) {
    return {
        {"run_type", jsonString("single")},
        {"test_case_id", std::to_string(selected_case.test_id)},
        {"seed", std::to_string(selected_case.seed)},
        {"device", jsonString(deviceTypeName(config.device))},
        {"num_macros", std::to_string(selected_case.num_macros)},
        {"num_std_cells", std::to_string(selected_case.num_std_cells)},
        {"total_cells", std::to_string(normalized_metrics.total_cells)},
        {"num_nets", std::to_string(normalized_metrics.num_nets)},
        {"initial_overlap_count",
         std::to_string(initial_metrics.overlap_count)},
        {"initial_total_overlap_area",
         jsonDouble(initial_metrics.total_overlap_area)},
        {"initial_max_overlap_area",
         jsonDouble(initial_metrics.max_overlap_area)},
        {"initial_overlap_percentage",
         jsonDouble(initial_metrics.overlap_percentage)},
        {"initial_cells_with_overlap",
         std::to_string(initial_metrics.cells_with_overlap)},
        {"initial_has_zero_overlap",
         jsonBool(initial_metrics.has_zero_overlap)},
        {"final_overlap_count", std::to_string(final_metrics.overlap_count)},
        {"final_total_overlap_area",
         jsonDouble(final_metrics.total_overlap_area)},
        {"final_max_overlap_area", jsonDouble(final_metrics.max_overlap_area)},
        {"final_overlap_percentage",
         jsonDouble(final_metrics.overlap_percentage)},
        {"final_cells_with_overlap",
         std::to_string(final_metrics.cells_with_overlap)},
        {"final_has_zero_overlap", jsonBool(final_metrics.has_zero_overlap)},
        {"normalized_overlap_ratio",
         jsonDouble(normalized_metrics.overlap_ratio)},
        {"normalized_wl", jsonDouble(normalized_metrics.normalized_wl)},
        {"normalized_num_cells_with_overlaps",
         std::to_string(normalized_metrics.num_cells_with_overlaps)},
        {"normalized_total_cells",
         std::to_string(normalized_metrics.total_cells)},
        {"normalized_num_nets", std::to_string(normalized_metrics.num_nets)},
        {"passed", jsonBool(passed)},
        {"stopped_early", jsonBool(training_result.stopped_early)},
        {"stop_reason", jsonString(training_result.stop_reason)},
        {"best_epoch", std::to_string(training_result.best_epoch)},
        {"epochs_completed", std::to_string(training_result.epochs_completed)},
        {"num_epochs", std::to_string(config.num_epochs)},
        {"early_stop_enabled", jsonBool(config.early_stop_enabled)},
        {"early_stop_patience", std::to_string(config.early_stop_patience)},
        {"early_stop_min_delta", jsonDouble(config.early_stop_min_delta)},
        {"early_stop_overlap_threshold",
         jsonDouble(config.early_stop_overlap_threshold)},
        {"early_stop_zero_overlap_patience",
         std::to_string(config.early_stop_zero_overlap_patience)},
        {"visualization_path", jsonString(visualization_path.string())},
    };
}

void writeSinglePlacementArtifacts(
    const CliOptions& options,
    const placement::BenchmarkCase& selected_case,
    const placement::TrainingConfig& config,
    const placement::OverlapMetrics& initial_metrics,
    const placement::OverlapMetrics& final_metrics,
    const placement::Metrics& normalized_metrics,
    const placement::TrainingResult& training_result,
    bool passed) {
    const std::filesystem::path visualization_path =
        outputFilePath(options, "placement_result.png");
    placement::plotPlacement(
        training_result.initial_cell_features,
        training_result.final_cell_features,
        visualization_path);

    writeCsvFile(
        outputFilePath(options, "placement_result_summary.csv"),
        singlePlacementHeader(),
        {singlePlacementRow(
            selected_case,
            config,
            initial_metrics,
            final_metrics,
            normalized_metrics,
            training_result,
            passed,
            visualization_path)});

    std::ostringstream json;
    appendJsonObject(
        json,
        singlePlacementJsonFields(
            selected_case,
            config,
            initial_metrics,
            final_metrics,
            normalized_metrics,
            training_result,
            passed,
            visualization_path),
        0);
    json << "\n";
    writeTextFile(outputFilePath(options, "placement_result_summary.json"), json.str());
}

void writeBenchmarkArtifacts(
    const CliOptions& options,
    const placement::BenchmarkSummary& summary) {
    std::vector<std::vector<std::string>> case_rows;
    case_rows.reserve(summary.results.size());
    for (const placement::BenchmarkResult& result : summary.results) {
        case_rows.push_back(benchmarkResultRow(result));
    }
    writeCsvFile(
        outputFilePath(options, "placement_benchmark_cases.csv"),
        benchmarkResultHeader(),
        case_rows);

    writeCsvFile(
        outputFilePath(options, "placement_benchmark_summary.csv"),
        {"total_cases",
         "average_overlap",
         "average_wirelength",
         "total_elapsed_seconds",
         "passed_count",
         "failed_count"},
        {{std::to_string(summary.results.size()),
          formatDouble(summary.average_overlap),
          formatDouble(summary.average_wirelength),
          formatDouble(summary.total_elapsed_seconds),
          std::to_string(summary.passed_count),
          std::to_string(summary.failed_count)}});

    std::ostringstream json;
    json << "{\n";
    appendJsonField(
        json,
        2,
        "total_cases",
        std::to_string(summary.results.size()),
        true);
    appendJsonField(
        json,
        2,
        "average_overlap",
        jsonDouble(summary.average_overlap),
        true);
    appendJsonField(
        json,
        2,
        "average_wirelength",
        jsonDouble(summary.average_wirelength),
        true);
    appendJsonField(
        json,
        2,
        "total_elapsed_seconds",
        jsonDouble(summary.total_elapsed_seconds),
        true);
    appendJsonField(
        json,
        2,
        "passed_count",
        std::to_string(summary.passed_count),
        true);
    appendJsonField(
        json,
        2,
        "failed_count",
        std::to_string(summary.failed_count),
        true);
    json << "  \"cases\": [\n";
    for (std::size_t index = 0; index < summary.results.size(); ++index) {
        appendJsonObject(json, benchmarkResultJsonFields(summary.results[index]), 4);
        if (index + 1 < summary.results.size()) {
            json << ",";
        }
        json << "\n";
    }
    json << "  ]\n";
    json << "}\n";
    writeTextFile(outputFilePath(options, "placement_benchmark_summary.json"), json.str());
}

void printRule(char c = '=') {
    std::cout << std::string(70, c) << "\n";
}

void printTrainingConfig(const placement::TrainingConfig& config) {
    std::cout << "Using hyperparameters:\n";
    std::cout << "  num_epochs: " << config.num_epochs << "\n";
    std::cout << "  lr: " << config.lr << "\n";
    std::cout << "  lambda_wirelength: " << config.lambda_wirelength << "\n";
    std::cout << "  lambda_overlap: " << config.lambda_overlap << "\n";
    std::cout << "  scheduler: " << config.scheduler_name << "\n";
    std::cout << "  scheduler_patience: " << config.scheduler_patience << "\n";
    std::cout << "  scheduler_factor: " << config.scheduler_factor << "\n";
    std::cout << "  scheduler_eta_min: " << config.scheduler_eta_min << "\n";
    std::cout << "  scheduler_step_size: " << config.scheduler_step_size << "\n";
    std::cout << "  scheduler_gamma: " << config.scheduler_gamma << "\n";
    std::cout << "  track_loss_history: "
              << (config.track_loss_history ? "true" : "false") << "\n";
    std::cout << "  track_overlap_metrics: "
              << (config.track_overlap_metrics ? "true" : "false") << "\n";
    std::cout << "  early_stop_enabled: "
              << (config.early_stop_enabled ? "true" : "false") << "\n";
    std::cout << "  early_stop_patience: " << config.early_stop_patience << "\n";
    std::cout << "  early_stop_min_delta: " << config.early_stop_min_delta << "\n";
    std::cout << "  early_stop_overlap_threshold: "
              << config.early_stop_overlap_threshold << "\n";
    std::cout << "  early_stop_zero_overlap_patience: "
              << config.early_stop_zero_overlap_patience << "\n";
}

void printBenchmarkResult(const placement::BenchmarkResult& result) {
    const char* status = result.passed ? "PASS" : "FAIL";
    std::cout << "Completed test " << result.test_id << ":\n";
    std::cout << "  Device: " << deviceTypeName(result.device) << "\n";
    std::cout << "  Overlap Ratio: " << std::fixed << std::setprecision(4)
              << result.overlap_ratio << " (" << result.num_cells_with_overlaps
              << "/" << result.total_cells << " cells)\n";
    std::cout << "  Normalized WL: " << std::fixed << std::setprecision(4)
              << result.normalized_wl << "\n";
    std::cout << "  Epochs Completed: " << result.epochs_completed << "\n";
    if (result.stopped_early) {
        std::cout << "  Stopped Early: " << result.stop_reason
                  << " at best epoch " << result.best_epoch << "\n";
    }
    std::cout << "  Time: " << std::fixed << std::setprecision(2)
              << result.elapsed_seconds << "s\n";
    std::cout << "  Status: " << status << "\n\n";
}

int runBenchmark(
    const CliOptions& options,
    const placement::TrainingConfig& config) {
    printRule();
    std::cout << "PLACEMENT CHALLENGE TEST SUITE\n";
    printRule();
    std::cout << "\nRunning " << placement::activeBenchmarkCases().size()
              << " active test cases serially.\n";
    printTrainingConfig(config);
    std::cout << "\n";

    std::optional<std::filesystem::path> loss_tracking_db_path;
    if (config.track_loss_history) {
        loss_tracking_db_path = createLossTrackingDb();
        std::cout << "Writing loss history to: "
                  << loss_tracking_db_path->string() << "\n\n";
    } else {
        std::cout << "Loss history tracking disabled.\n\n";
    }

    int case_index = 1;
    for (const placement::BenchmarkCase& test_case :
         placement::activeBenchmarkCases()) {
        const char* size_category =
            test_case.num_std_cells <= 30
                ? "Small"
                : test_case.num_std_cells <= 100 ? "Medium" : "Large";
        std::cout << "Test " << case_index++ << "/"
                  << placement::activeBenchmarkCases().size() << ": "
                  << size_category << " (" << test_case.num_macros
                  << " macros, " << test_case.num_std_cells << " std cells)\n";
        std::cout << "  Seed: " << test_case.seed << "\n";
    }
    std::cout << "\n";

    const placement::BenchmarkSummary summary =
        placement::runActiveBenchmarkCases(config);
    for (const placement::BenchmarkResult& result : summary.results) {
        printBenchmarkResult(result);
    }

    if (loss_tracking_db_path.has_value()) {
        for (const placement::BenchmarkResult& result : summary.results) {
            saveLossHistorySqlite(
                result.loss_history,
                *loss_tracking_db_path,
                LossHistoryRunMetadata{
                    .test_id = result.test_id,
                    .runner = "placement.cpp --benchmark",
                    .run_label = "train_placement",
                    .run_started_at = result.run_started_at,
                    .seed = result.seed,
                    .num_macros = result.num_macros,
                    .num_std_cells = result.num_std_cells,
                    .num_epochs = config.num_epochs,
                    .lr = config.lr,
                    .lambda_wirelength = config.lambda_wirelength,
                    .lambda_overlap = config.lambda_overlap,
                    .log_interval = config.log_interval,
                    .verbose = config.verbose,
                    .total_cells = result.total_cells,
                    .total_pins = result.total_pins,
                    .total_edges = result.num_nets,
                });
        }
        std::cout << "Loss history saved to: "
                  << loss_tracking_db_path->string() << "\n\n";
    }

    printRule();
    std::cout << "FINAL RESULTS\n";
    printRule();
    std::cout << "Average Overlap: " << std::fixed << std::setprecision(4)
              << summary.average_overlap << "\n";
    std::cout << "Average Wirelength: " << std::fixed << std::setprecision(4)
              << summary.average_wirelength << "\n";
    std::cout << "Total Runtime: " << std::fixed << std::setprecision(2)
              << summary.total_elapsed_seconds << "s\n";
    std::cout << "Passed: " << summary.passed_count << "\n";
    std::cout << "Failed: " << summary.failed_count << "\n";
    if (options.write_output_files) {
        writeBenchmarkArtifacts(options, summary);
    }
    return 0;
}

int runSinglePlacement(
    const CliOptions& options,
    const placement::TrainingConfig& config) {
    placement::BenchmarkCase selected_case{
        0,
        options.num_macros,
        options.num_std_cells,
        options.seed,
    };
    if (options.test_case_id != 0) {
        const std::optional<placement::BenchmarkCase> test_case =
            findBenchmarkCase(options.test_case_id);
        if (!test_case.has_value()) {
            throw std::invalid_argument(
                "Unknown benchmark test case id: " +
                std::to_string(options.test_case_id));
        }
        selected_case = *test_case;
    }

    printRule();
    std::cout << "VLSI CELL PLACEMENT OPTIMIZATION\n";
    printRule();
    std::cout << "\nGenerating placement problem:\n";
    if (selected_case.test_id != 0) {
        std::cout << "  - benchmark test case: " << selected_case.test_id << "\n";
    }
    std::cout << "  - " << selected_case.num_macros << " macros\n";
    std::cout << "  - " << selected_case.num_std_cells << " standard cells\n";
    std::cout << "  - seed: " << selected_case.seed << "\n";
    std::cout << "  - device: " << deviceTypeName(config.device) << "\n";

    seedTorch(selected_case.seed);
    const torch::Device device(config.device);
    placement::PlacementProblem problem = placement::generatePlacementInput(
        selected_case.num_macros,
        selected_case.num_std_cells,
        device);
    placement::initializeCellPositions(problem.cell_features);

    std::cout << "\n";
    printRule();
    std::cout << "INITIAL STATE\n";
    printRule();
    const placement::OverlapMetrics initial_metrics =
        placement::calculateOverlapMetrics(problem.cell_features);
    std::cout << "Overlap count: " << initial_metrics.overlap_count << "\n";
    std::cout << "Total overlap area: " << std::fixed << std::setprecision(2)
              << initial_metrics.total_overlap_area << "\n";
    std::cout << "Max overlap area: " << std::fixed << std::setprecision(2)
              << initial_metrics.max_overlap_area << "\n";
    std::cout << "Overlap percentage: " << std::fixed << std::setprecision(2)
              << initial_metrics.overlap_percentage << "%\n";

    std::cout << "\n";
    printRule();
    std::cout << "RUNNING OPTIMIZATION\n";
    printRule();
    std::optional<std::filesystem::path> loss_tracking_db_path;
    if (config.track_loss_history) {
        loss_tracking_db_path = createLossTrackingDb();
        std::cout << "Writing loss history to: "
                  << loss_tracking_db_path->string() << "\n";
    } else {
        std::cout << "Loss history tracking disabled.\n";
    }

    placement::TrainingResult training_result = placement::trainPlacement(
        problem.cell_features,
        problem.pin_features,
        problem.edge_list,
        config);
    if (loss_tracking_db_path.has_value()) {
        const std::filesystem::path saved_path = saveLossHistorySqlite(
            training_result.loss_history,
            *loss_tracking_db_path,
            LossHistoryRunMetadata{
                .test_id = selected_case.test_id == 0
                    ? std::optional<int>()
                    : std::optional<int>(selected_case.test_id),
                .runner = "placement.cpp",
                .run_label = "train_placement",
                .run_started_at = training_result.run_started_at,
                .seed = selected_case.seed,
                .num_macros = selected_case.num_macros,
                .num_std_cells = selected_case.num_std_cells,
                .num_epochs = config.num_epochs,
                .lr = config.lr,
                .lambda_wirelength = config.lambda_wirelength,
                .lambda_overlap = config.lambda_overlap,
                .log_interval = config.log_interval,
                .verbose = config.verbose,
                .total_cells = problem.cell_features.size(0),
                .total_pins = problem.pin_features.size(0),
                .total_edges = problem.edge_list.size(0),
            });
        std::cout << "Loss history saved to: " << saved_path.string() << "\n";
    }

    std::cout << "\n";
    printRule();
    std::cout << "FINAL RESULTS\n";
    printRule();
    const placement::OverlapMetrics final_overlap_metrics =
        placement::calculateOverlapMetrics(training_result.final_cell_features);
    std::cout << "Overlap count (pairs): "
              << final_overlap_metrics.overlap_count << "\n";
    std::cout << "Total overlap area: " << std::fixed << std::setprecision(2)
              << final_overlap_metrics.total_overlap_area << "\n";
    std::cout << "Max overlap area: " << std::fixed << std::setprecision(2)
              << final_overlap_metrics.max_overlap_area << "\n";

    std::cout << "\n";
    printRule('-');
    std::cout << "TEST SUITE METRICS\n";
    printRule('-');
    const placement::Metrics normalized_metrics =
        placement::calculateNormalizedMetrics(
            training_result.final_cell_features,
            problem.pin_features,
            problem.edge_list);
    std::cout << "Overlap Ratio: " << std::fixed << std::setprecision(4)
              << normalized_metrics.overlap_ratio << " ("
              << normalized_metrics.num_cells_with_overlaps << "/"
              << normalized_metrics.total_cells << " cells)\n";
    std::cout << "Normalized Wirelength: " << std::fixed << std::setprecision(4)
              << normalized_metrics.normalized_wl << "\n";
    std::cout << "Epochs Completed: " << training_result.epochs_completed << "\n";
    if (training_result.stopped_early) {
        std::cout << "Stopped Early: " << training_result.stop_reason
                  << " at best epoch " << training_result.best_epoch << "\n";
    }

    std::cout << "\n";
    printRule();
    std::cout << "SUCCESS CRITERIA\n";
    printRule();
    const bool passed = normalized_metrics.num_cells_with_overlaps == 0;
    if (passed) {
        std::cout << "PASS: No overlapping cells.\n";
    } else {
        std::cout << "FAIL: Overlaps remain in "
                  << normalized_metrics.num_cells_with_overlaps << " cells.\n";
    }

    if (options.write_output_files) {
        writeSinglePlacementArtifacts(
            options,
            selected_case,
            config,
            initial_metrics,
            final_overlap_metrics,
            normalized_metrics,
            training_result,
            passed);
    }

    return 0;
}

void configureCli(
    CLI::App& app,
    CliOptions& options,
    placement::TrainingConfig& config) {
    app.add_flag(
        "--benchmark",
        options.run_benchmark,
        "Run the active benchmark suite instead of a single placement.");
    app.add_option(
           "--device",
           options.device,
           "Device to run on: auto, cpu, cuda, or mps.")
        ->check(CLI::IsMember({"auto", "cpu", "cuda", "mps"}));
    app.add_option(
        "--test-case-id",
        options.test_case_id,
        "Optional benchmark test case id for a single placement run.");
    app.add_option(
        "--num-macros",
        options.num_macros,
        "Number of macro cells for a single placement run.");
    app.add_option(
        "--num-std-cells",
        options.num_std_cells,
        "Number of standard cells for a single placement run.");
    app.add_option("--seed", options.seed, "Random seed for a single placement run.");
    app.add_flag(
        "--write-output-files",
        options.write_output_files,
        "Write notebook-friendly CSV and JSON output artifacts.");
    app.add_option(
        "--output-dir",
        options.output_dir,
        "Directory for output artifacts when --write-output-files is set.");

    app.add_option(
        "--num-epochs",
        config.num_epochs,
        "Number of optimization epochs.");
    app.add_option("--lr", config.lr, "Learning rate for Adam.");
    app.add_option(
        "--lambda-wirelength",
        config.lambda_wirelength,
        "Weight applied to the wirelength loss.");
    app.add_option(
        "--lambda-overlap",
        config.lambda_overlap,
        "Weight applied to the overlap loss.");
    app.add_option("--scheduler", config.scheduler_name, "Learning-rate scheduler.")
        ->check(CLI::IsMember(
            {"plateau", "cosine", "step", "exponential", "none"}));
    app.add_option(
        "--scheduler-patience",
        config.scheduler_patience,
        "Patience for ReduceLROnPlateau.");
    app.add_option(
        "--scheduler-factor",
        config.scheduler_factor,
        "Decay factor for ReduceLROnPlateau.");
    app.add_option(
        "--scheduler-eta-min",
        config.scheduler_eta_min,
        "Minimum learning rate for cosine annealing.");
    app.add_option(
        "--scheduler-step-size",
        config.scheduler_step_size,
        "Step size in epochs for StepLR.");
    app.add_option(
        "--scheduler-gamma",
        config.scheduler_gamma,
        "Gamma decay for step and exponential schedulers.");
    app.add_flag(
        "--track-loss-history",
        [&config](int64_t count) {
            if (count > 0) {
                config.track_loss_history = true;
            }
        },
        "Collect and persist per-epoch loss history to SQLite.");
    app.add_flag(
        "--no-track-loss-history",
        [&config](int64_t count) {
            if (count > 0) {
                config.track_loss_history = false;
            }
        },
        "Disable per-epoch loss-history persistence.");
    app.add_flag(
        "--track-overlap-metrics",
        [&config](int64_t count) {
            if (count > 0) {
                config.track_overlap_metrics = true;
            }
        },
        "Compute overlap metrics every epoch.");
    app.add_flag(
        "--no-track-overlap-metrics",
        [&config](int64_t count) {
            if (count > 0) {
                config.track_overlap_metrics = false;
            }
        },
        "Disable per-epoch overlap-metric tracking.");
    app.add_flag(
        "--no-early-stop",
        [&config](int64_t count) {
            if (count > 0) {
                config.early_stop_enabled = false;
            }
        },
        "Disable overlap-first early stopping.");
    app.add_option(
        "--early-stop-patience",
        config.early_stop_patience,
        "Patience before stopping when overlap stops improving.");
    app.add_option(
        "--early-stop-min-delta",
        config.early_stop_min_delta,
        "Minimum improvement required to reset early-stop patience.");
    app.add_option(
        "--early-stop-overlap-threshold",
        config.early_stop_overlap_threshold,
        "Overlap threshold treated as effectively zero.");
    app.add_option(
        "--early-stop-zero-overlap-patience",
        config.early_stop_zero_overlap_patience,
        "Extra patience after zero overlap is reached.");
    app.add_flag("--quiet", [&config](int64_t count) {
        if (count > 0) {
            config.verbose = false;
        }
    }, "Suppress per-epoch output for a single placement run.");
    app.add_option(
        "--log-interval",
        config.log_interval,
        "Epoch interval for verbose training logs.");
}

void validateOptions(
    const CliOptions& options,
    const placement::TrainingConfig& config) {
    if (options.num_macros < 0 || options.num_std_cells < 0) {
        throw std::invalid_argument("Cell counts must be nonnegative");
    }
    if (options.num_macros + options.num_std_cells < 0) {
        throw std::invalid_argument("Cell counts overflowed");
    }
    if (config.num_epochs < 0) {
        throw std::invalid_argument("Number of epochs must be nonnegative");
    }
    if (config.lr <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    if (config.scheduler_patience < 0) {
        throw std::invalid_argument("Scheduler patience must be nonnegative");
    }
    if (config.scheduler_factor <= 0.0) {
        throw std::invalid_argument("Scheduler factor must be positive");
    }
    if (config.scheduler_step_size <= 0) {
        throw std::invalid_argument("Scheduler step size must be positive");
    }
    if (config.early_stop_patience <= 0 ||
        config.early_stop_zero_overlap_patience <= 0) {
        throw std::invalid_argument("Early-stop patience values must be positive");
    }
}

}  // namespace

int main(int argc, char** argv) {
    CliOptions options;
    placement::TrainingConfig config;
    config.log_interval = 200;

    CLI::App app{"Placement C++ runner"};
    configureCli(app, options, config);
    CLI11_PARSE(app, argc, argv);

    try {
        validateOptions(options, config);
        config.device = resolveDeviceType(options.device);

        if (options.run_benchmark) {
            config.verbose = false;
            return runBenchmark(options, config);
        }
        return runSinglePlacement(options, config);
    } catch (const c10::Error& error) {
        std::cerr << "LibTorch error: " << error.what_without_backtrace() << "\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
    }
    return 1;
}
