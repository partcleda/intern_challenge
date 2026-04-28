#include "placement/benchmark.h"

#include "placement/generation.h"
#include "placement/metrics.h"
#include "placement/training.h"

#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace placement {

namespace {

PlacementProblem generateSeededProblem(
    const BenchmarkCase& test_case,
    const TrainingConfig& benchmark_config,
    std::mutex* rng_mutex) {
    const auto generate = [&]() {
        if (test_case.seed != 0) {
            torch::manual_seed(test_case.seed);
        }

        const torch::Device device(benchmark_config.device);
        PlacementProblem problem = generatePlacementInput(
            test_case.num_macros,
            test_case.num_std_cells,
            device,
            false);
        initializeCellPositions(problem.cell_features);
        return problem;
    };

    if (rng_mutex == nullptr) {
        return generate();
    }

    std::lock_guard<std::mutex> lock(*rng_mutex);
    return generate();
}

BenchmarkResult runBenchmarkCaseImpl(
    const BenchmarkCase& test_case,
    const TrainingConfig& config,
    std::mutex* rng_mutex) {
    TrainingConfig benchmark_config = config;
    benchmark_config.verbose = false;

    PlacementProblem problem =
        generateSeededProblem(test_case, benchmark_config, rng_mutex);

    const auto start_time = std::chrono::steady_clock::now();
    const TrainingResult training_result = trainPlacement(
        problem.cell_features,
        problem.pin_features,
        problem.edge_list,
        benchmark_config);
    const auto elapsed = std::chrono::steady_clock::now() - start_time;

    const Metrics metrics = calculateNormalizedMetrics(
        training_result.final_cell_features,
        problem.pin_features,
        problem.edge_list);

    BenchmarkResult result;
    result.test_id = test_case.test_id;
    result.num_macros = test_case.num_macros;
    result.num_std_cells = test_case.num_std_cells;
    result.total_cells = metrics.total_cells;
    result.total_pins = problem.pin_features.size(0);
    result.num_nets = metrics.num_nets;
    result.seed = test_case.seed;
    result.device = benchmark_config.device;
    result.elapsed_seconds = std::chrono::duration<double>(elapsed).count();
    result.num_cells_with_overlaps = metrics.num_cells_with_overlaps;
    result.overlap_ratio = metrics.overlap_ratio;
    result.normalized_wl = metrics.normalized_wl;
    result.passed = result.num_cells_with_overlaps == 0;
    result.stopped_early = training_result.stopped_early;
    result.stop_reason = training_result.stop_reason;
    result.run_started_at = training_result.run_started_at;
    result.best_epoch = training_result.best_epoch;
    result.epochs_completed = training_result.epochs_completed;
    result.loss_history = training_result.loss_history;
    return result;
}

void addResultToSummary(BenchmarkSummary& summary, BenchmarkResult result) {
    if (result.passed) {
        ++summary.passed_count;
    } else {
        ++summary.failed_count;
    }
    summary.results.push_back(std::move(result));
}

void finalizeSummary(BenchmarkSummary& summary) {
    if (summary.results.empty()) {
        return;
    }

    double overlap_sum = 0.0;
    double wirelength_sum = 0.0;
    for (const BenchmarkResult& result : summary.results) {
        overlap_sum += result.overlap_ratio;
        wirelength_sum += result.normalized_wl;
    }

    const double case_count = static_cast<double>(summary.results.size());
    summary.average_overlap = overlap_sum / case_count;
    summary.average_wirelength = wirelength_sum / case_count;
}

}  // namespace

const std::vector<BenchmarkCase>& activeBenchmarkCases() {
    static const std::vector<BenchmarkCase> cases = {
        {1, 2, 20, 1001},
        {2, 3, 25, 1002},
        {3, 2, 30, 1003},
        {4, 3, 50, 1004},
        {5, 4, 75, 1005},
        {6, 5, 100, 1006},
        {7, 5, 150, 1007},
        {8, 7, 150, 1008},
        {9, 8, 200, 1009},
        {10, 10, 2000, 1010},
    };
    return cases;
}

BenchmarkResult runBenchmarkCase(
    const BenchmarkCase& test_case,
    const TrainingConfig& config) {
    return runBenchmarkCaseImpl(test_case, config, nullptr);
}

BenchmarkSummary runBenchmarkCases(
    const std::vector<BenchmarkCase>& test_cases,
    const TrainingConfig& config,
    int worker_count) {
    if (worker_count <= 0) {
        throw std::invalid_argument("Worker count must be positive");
    }

    BenchmarkSummary summary;
    if (test_cases.empty()) {
        return summary;
    }

    summary.results.reserve(test_cases.size());
    const auto start_time = std::chrono::steady_clock::now();

    if (worker_count == 1) {
        for (const BenchmarkCase& test_case : test_cases) {
            addResultToSummary(
                summary,
                runBenchmarkCaseImpl(test_case, config, nullptr));
        }
    } else {
        std::vector<BenchmarkResult> ordered_results(test_cases.size());
        std::atomic<std::size_t> next_index{0};
        std::atomic<bool> should_stop{false};
        std::mutex rng_mutex;
        std::mutex exception_mutex;
        std::exception_ptr first_exception;

        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(worker_count));
        for (int worker_index = 0; worker_index < worker_count; ++worker_index) {
            workers.emplace_back([&]() {
                while (!should_stop.load()) {
                    const std::size_t index = next_index.fetch_add(1);
                    if (index >= test_cases.size()) {
                        return;
                    }

                    try {
                        ordered_results[index] = runBenchmarkCaseImpl(
                            test_cases[index],
                            config,
                            &rng_mutex);
                    } catch (...) {
                        {
                            std::lock_guard<std::mutex> lock(exception_mutex);
                            if (first_exception == nullptr) {
                                first_exception = std::current_exception();
                            }
                        }
                        should_stop.store(true);
                        return;
                    }
                }
            });
        }

        for (std::thread& worker : workers) {
            worker.join();
        }

        if (first_exception != nullptr) {
            std::rethrow_exception(first_exception);
        }

        for (BenchmarkResult& result : ordered_results) {
            addResultToSummary(summary, std::move(result));
        }
    }

    const auto elapsed = std::chrono::steady_clock::now() - start_time;

    finalizeSummary(summary);
    summary.total_elapsed_seconds = std::chrono::duration<double>(elapsed).count();

    return summary;
}

BenchmarkSummary runActiveBenchmarkCases(
    const TrainingConfig& config,
    int worker_count) {
    return runBenchmarkCases(activeBenchmarkCases(), config, worker_count);
}

}  // namespace placement
