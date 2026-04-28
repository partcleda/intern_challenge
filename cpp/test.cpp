#include "placement/benchmark.h"
#include "placement/types.h"

#include <CLI/CLI.hpp>
#include <torch/cuda.h>
#include <torch/mps.h>
#include <torch/torch.h>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct TestOptions {
    std::string device = "auto";
    int workers = 1;
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

void printRule() {
    std::cout << std::string(70, '=') << "\n";
}

void printTrainingConfig(const placement::TrainingConfig& config, int workers) {
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
    std::cout << "  workers: " << workers << "\n";
}

const char* sizeCategory(const placement::BenchmarkCase& test_case) {
    if (test_case.num_std_cells <= 30) {
        return "Small";
    }
    if (test_case.num_std_cells <= 100) {
        return "Medium";
    }
    return "Large";
}

void printCaseList() {
    int case_index = 1;
    const std::vector<placement::BenchmarkCase>& cases =
        placement::activeBenchmarkCases();
    for (const placement::BenchmarkCase& test_case : cases) {
        std::cout << "Test " << case_index++ << "/" << cases.size() << ": "
                  << sizeCategory(test_case) << " (" << test_case.num_macros
                  << " macros, " << test_case.num_std_cells << " std cells)\n";
        std::cout << "  Seed: " << test_case.seed << "\n";
    }
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

void configureCli(
    CLI::App& app,
    TestOptions& options,
    placement::TrainingConfig& config) {
    app.add_option(
           "--device",
           options.device,
           "Device to run on: auto, cpu, cuda, or mps.")
        ->check(CLI::IsMember({"auto", "cpu", "cuda", "mps"}));
    app.add_option(
        "--workers",
        options.workers,
        "Number of worker threads for benchmark cases.");
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
        "--track-overlap-metrics",
        config.track_overlap_metrics,
        "Compute overlap metrics every epoch.");
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
}

void validateOptions(
    const TestOptions& options,
    const placement::TrainingConfig& config) {
    if (options.workers <= 0) {
        throw std::invalid_argument("Worker count must be positive");
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

int runTestSuite(
    const TestOptions& options,
    const placement::TrainingConfig& config,
    const char* binary_path) {
    printRule();
    std::cout << "PLACEMENT CHALLENGE TEST SUITE\n";
    printRule();
    std::cout << "\nBinary: " << std::filesystem::absolute(binary_path).string()
              << "\n";
    std::cout << "\nRunning " << placement::activeBenchmarkCases().size()
              << " test cases with various netlist sizes...\n";
    printTrainingConfig(config, options.workers);
    std::cout << "\nLoss history tracking disabled.\n\n";
    printCaseList();
    if (options.workers == 1) {
        std::cout << "Running serially\n\n";
    } else {
        std::cout << "Running with " << options.workers << " workers\n\n";
    }

    const placement::BenchmarkSummary summary =
        placement::runActiveBenchmarkCases(config, options.workers);
    for (const placement::BenchmarkResult& result : summary.results) {
        printBenchmarkResult(result);
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
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    TestOptions options;
    placement::TrainingConfig config;
    config.verbose = false;

    CLI::App app{"Placement C++ test suite runner"};
    configureCli(app, options, config);
    CLI11_PARSE(app, argc, argv);

    try {
        validateOptions(options, config);
        config.device = resolveDeviceType(options.device);
        return runTestSuite(options, config, argv[0]);
    } catch (const c10::Error& error) {
        std::cerr << "LibTorch error: " << error.what_without_backtrace() << "\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
    }
    return 1;
}
