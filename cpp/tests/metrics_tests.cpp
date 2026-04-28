#include "placement/benchmark.h"
#include "placement/generation.h"
#include "placement/losses.h"
#include "placement/metrics.h"
#include "placement/training.h"

#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void visualizationWritesPngWithExpectedContent();

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expectNear(
    double actual,
    double expected,
    double tolerance,
    const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
    }
}

placement::TrainingConfig fastBenchmarkConfig() {
    placement::TrainingConfig config;
    config.device = torch::kCPU;
    config.num_epochs = 0;
    config.scheduler_name = "none";
    config.early_stop_enabled = false;
    config.verbose = false;
    return config;
}

void deterministicMetricsMatchPythonReference() {
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64);

    const auto cell_features = torch::tensor(
        {
            {4.0F, 1.0F, 0.0F, 0.0F, 2.0F, 2.0F},
            {4.0F, 1.0F, 1.0F, 0.0F, 2.0F, 2.0F},
            {1.0F, 1.0F, 10.0F, 10.0F, 1.0F, 1.0F},
        },
        float_options);
    const auto pin_features = torch::tensor(
        {
            {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F},
            {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F},
            {2.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F},
        },
        float_options);
    const auto edge_list = torch::tensor({{0LL, 2LL}}, long_options);

    const placement::OverlapMetrics overlap =
        placement::calculateOverlapMetrics(cell_features);
    expect(overlap.overlap_count == 1, "expected one overlapping pair");
    expectNear(overlap.total_overlap_area, 2.0, 1e-5, "total overlap area");
    expectNear(overlap.max_overlap_area, 2.0, 1e-5, "max overlap area");
    expectNear(overlap.overlap_percentage, 100.0 / 3.0, 1e-5, "overlap percentage");
    expect(overlap.cells_with_overlap == 2, "expected two cells with overlap");
    expect(!overlap.has_zero_overlap, "expected nonzero overlap flag");

    const placement::Metrics metrics =
        placement::calculateNormalizedMetrics(cell_features, pin_features, edge_list);
    expect(metrics.total_cells == 3, "expected three total cells");
    expect(metrics.num_nets == 1, "expected one net");
    expect(metrics.num_cells_with_overlaps == 2, "expected two overlapping cells");
    expectNear(metrics.overlap_ratio, 2.0 / 3.0, 1e-6, "overlap ratio");

    const double smooth_wirelength = 10.0 + 0.1 * std::log(2.0);
    expectNear(
        metrics.normalized_wl,
        smooth_wirelength / 3.0,
        1e-5,
        "normalized wirelength");

    const auto no_edges = torch::zeros({0, 2}, long_options);
    const placement::Metrics no_edge_metrics =
        placement::calculateNormalizedMetrics(cell_features, pin_features, no_edges);
    expectNear(no_edge_metrics.normalized_wl, 0.0, 1e-12, "zero-edge wirelength");
}

void generatedProblemCanBeMeasured() {
    torch::manual_seed(66);
    placement::PlacementProblem problem =
        placement::generatePlacementInput(2, 5, torch::kCPU, false);

    expect(problem.cell_features.size(0) == 7, "generated cell count");
    expect(problem.cell_features.size(1) == 6, "generated cell feature width");
    expect(problem.pin_features.size(1) == 7, "generated pin feature width");
    expect(problem.edge_list.size(1) == 2, "generated edge width");

    placement::initializeCellPositions(problem.cell_features);
    const placement::Metrics metrics = placement::calculateNormalizedMetrics(
        problem.cell_features,
        problem.pin_features,
        problem.edge_list);

    expect(metrics.total_cells == 7, "metrics total cell count");
    expect(metrics.num_nets == problem.edge_list.size(0), "metrics net count");
    expect(
        metrics.num_cells_with_overlaps >= 0 &&
            metrics.num_cells_with_overlaps <= metrics.total_cells,
        "overlapping cell count range");
    expect(
        metrics.overlap_ratio >= 0.0 && metrics.overlap_ratio <= 1.0,
        "overlap ratio range");
    expect(std::isfinite(metrics.normalized_wl), "finite normalized wirelength");
    expect(metrics.normalized_wl >= 0.0, "nonnegative normalized wirelength");
}

void deterministicLossesMatchPythonReference() {
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64);

    const auto cell_features = torch::tensor(
        {
            {4.0F, 1.0F, 0.0F, 0.0F, 2.0F, 2.0F},
            {4.0F, 1.0F, 1.0F, 0.0F, 2.0F, 2.0F},
            {1.0F, 1.0F, 10.0F, 10.0F, 1.0F, 1.0F},
        },
        float_options);
    const auto pin_features = torch::tensor(
        {
            {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F},
            {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F},
            {2.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F},
        },
        float_options);
    const auto edge_list = torch::tensor({{0LL, 2LL}}, long_options);

    const auto pairwise_overlap =
        placement::computePairwiseOverlapAreas(cell_features);
    const auto expected_pairwise_overlap = torch::tensor(
        {
            {4.0F, 2.0F, 0.0F},
            {2.0F, 4.0F, 0.0F},
            {0.0F, 0.0F, 1.0F},
        },
        float_options);
    expect(
        torch::allclose(pairwise_overlap, expected_pairwise_overlap),
        "pairwise overlap areas");

    const double smooth_wirelength = 10.0 + 0.1 * std::log(2.0);
    expectNear(
        placement::wirelengthAttractionLoss(cell_features, pin_features, edge_list)
            .item<double>(),
        smooth_wirelength,
        1e-5,
        "wirelength attraction loss");

    expectNear(
        placement::overlapRepulsionLoss(cell_features, pin_features, edge_list)
            .item<double>(),
        std::log1p(2.0) * 200.0,
        1e-4,
        "overlap repulsion loss");
}

void lossEdgeCasesStayFinite() {
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64);

    const auto single_cell = torch::tensor(
        {{4.0F, 1.0F, 0.0F, 0.0F, 2.0F, 2.0F}},
        float_options);
    const auto single_pin = torch::tensor(
        {{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F}},
        float_options);
    const auto no_edges = torch::zeros({0, 2}, long_options);

    const auto single_overlap =
        placement::computePairwiseOverlapAreas(single_cell);
    expect(single_overlap.size(0) == 1, "single-cell overlap row count");
    expect(single_overlap.size(1) == 1, "single-cell overlap column count");
    expectNear(
        placement::wirelengthAttractionLoss(single_cell, single_pin, no_edges)
            .item<double>(),
        0.0,
        1e-12,
        "zero-edge wirelength loss");
    expectNear(
        placement::overlapRepulsionLoss(single_cell, single_pin, no_edges)
            .item<double>(),
        0.0,
        1e-12,
        "single-cell overlap loss");
}

void lossesBackpropagateThroughCellPositions() {
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64);

    const auto cell_static_prefix = torch::tensor(
        {
            {4.0F, 1.0F},
            {4.0F, 1.0F},
        },
        float_options);
    auto positions = torch::tensor(
        {
            {0.0F, 0.0F},
            {1.0F, 0.0F},
        },
        float_options.requires_grad(true));
    const auto cell_static_suffix = torch::tensor(
        {
            {2.0F, 2.0F},
            {2.0F, 2.0F},
        },
        float_options);
    const auto cell_features =
        torch::cat({cell_static_prefix, positions, cell_static_suffix}, 1);
    const auto pin_features = torch::tensor(
        {
            {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F},
            {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F},
        },
        float_options);
    const auto edge_list = torch::tensor({{0LL, 1LL}}, long_options);

    const auto loss =
        placement::wirelengthAttractionLoss(cell_features, pin_features, edge_list) +
        placement::overlapRepulsionLoss(cell_features, pin_features, edge_list);
    loss.backward();

    expect(positions.grad().defined(), "positions gradient is defined");
    expect(torch::all(torch::isfinite(positions.grad())).item<bool>(), "finite gradients");
    expect(positions.grad().abs().sum().item<double>() > 0.0, "nonzero gradients");
}

void trainingWithNoEpochsReturnsInitialPlacement() {
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64);

    const auto cell_features = torch::tensor(
        {
            {1.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F},
            {1.0F, 1.0F, 4.0F, 0.0F, 1.0F, 1.0F},
        },
        float_options);
    const auto pin_features = torch::tensor(
        {
            {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F},
            {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F},
        },
        float_options);
    const auto edge_list = torch::tensor({{0LL, 1LL}}, long_options);

    placement::TrainingConfig config;
    config.num_epochs = 0;
    config.verbose = false;

    const placement::TrainingResult result =
        placement::trainPlacement(cell_features, pin_features, edge_list, config);
    expect(
        torch::allclose(result.initial_cell_features, cell_features),
        "zero-epoch initial features");
    expect(
        torch::allclose(result.final_cell_features, cell_features),
        "zero-epoch final features");
    expect(!result.stopped_early, "zero-epoch does not stop early");
    expect(result.best_epoch == -1, "zero-epoch best epoch");
    expect(result.epochs_completed == 0, "zero-epoch epochs completed");
}

void trainingReducesOverlapLoss() {
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64);

    const auto cell_features = torch::tensor(
        {
            {4.0F, 1.0F, 0.0F, 0.0F, 2.0F, 2.0F},
            {4.0F, 1.0F, 1.0F, 0.0F, 2.0F, 2.0F},
        },
        float_options);
    const auto pin_features = torch::zeros({0, 7}, float_options);
    const auto edge_list = torch::zeros({0, 2}, long_options);

    placement::TrainingConfig config;
    config.num_epochs = 40;
    config.lr = 0.1;
    config.lambda_wirelength = 0.0;
    config.lambda_overlap = 1.0;
    config.scheduler_name = "none";
    config.early_stop_enabled = false;
    config.verbose = false;

    const double initial_overlap =
        placement::calculateOverlapMetrics(cell_features).total_overlap_area;
    const placement::TrainingResult result =
        placement::trainPlacement(cell_features, pin_features, edge_list, config);
    const double final_overlap =
        placement::calculateOverlapMetrics(result.final_cell_features)
            .total_overlap_area;

    expect(final_overlap < initial_overlap, "training reduces overlap area");
    expect(
        torch::allclose(result.initial_cell_features, cell_features),
        "training preserves initial features");
    expect(!result.stopped_early, "overlap-only training no early stop");
    expect(
        result.epochs_completed == config.num_epochs,
        "overlap-only training epochs completed");
}

void trainingReducesWirelengthLoss() {
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64);

    const auto cell_features = torch::tensor(
        {
            {1.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F},
            {1.0F, 1.0F, 10.0F, 0.0F, 1.0F, 1.0F},
        },
        float_options);
    const auto pin_features = torch::tensor(
        {
            {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F},
            {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F},
        },
        float_options);
    const auto edge_list = torch::tensor({{0LL, 1LL}}, long_options);

    placement::TrainingConfig config;
    config.num_epochs = 20;
    config.lr = 0.1;
    config.lambda_wirelength = 1.0;
    config.lambda_overlap = 0.0;
    config.scheduler_name = "none";
    config.early_stop_enabled = false;
    config.verbose = false;

    const double initial_wl =
        placement::wirelengthAttractionLoss(cell_features, pin_features, edge_list)
            .item<double>();
    const placement::TrainingResult result =
        placement::trainPlacement(cell_features, pin_features, edge_list, config);
    const double final_wl = placement::wirelengthAttractionLoss(
                                result.final_cell_features,
                                pin_features,
                                edge_list)
                                .item<double>();

    expect(final_wl < initial_wl, "training reduces wirelength");
    expect(
        result.epochs_completed == config.num_epochs,
        "wirelength training epochs completed");
}

void trainingReportsEarlyStopMetadata() {
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64);

    const auto cell_features = torch::tensor(
        {
            {1.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F},
            {1.0F, 1.0F, 4.0F, 0.0F, 1.0F, 1.0F},
        },
        float_options);
    const auto pin_features = torch::zeros({0, 7}, float_options);
    const auto edge_list = torch::zeros({0, 2}, long_options);

    placement::TrainingConfig config;
    config.num_epochs = 5;
    config.lr = 0.1;
    config.lambda_wirelength = 0.0;
    config.lambda_overlap = 1.0;
    config.scheduler_name = "none";
    config.early_stop_enabled = true;
    config.early_stop_zero_overlap_patience = 1;
    config.verbose = false;

    const placement::TrainingResult result =
        placement::trainPlacement(cell_features, pin_features, edge_list, config);

    expect(result.stopped_early, "training reports early stop");
    expect(
        result.stop_reason == "zero_overlap_plateau",
        "training early stop reason");
    expect(result.best_epoch == 0, "training best epoch");
    expect(result.epochs_completed == 2, "early-stop epochs completed");
}

void activeBenchmarkCasesMatchPythonReference() {
    const std::vector<placement::BenchmarkCase> expected = {
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

    const std::vector<placement::BenchmarkCase>& actual =
        placement::activeBenchmarkCases();
    expect(actual.size() == expected.size(), "active benchmark case count");

    for (std::size_t i = 0; i < expected.size(); ++i) {
        expect(actual[i].test_id == expected[i].test_id, "benchmark test id");
        expect(actual[i].num_macros == expected[i].num_macros, "benchmark macros");
        expect(
            actual[i].num_std_cells == expected[i].num_std_cells,
            "benchmark standard cells");
        expect(actual[i].seed == expected[i].seed, "benchmark seed");
    }
}

void benchmarkCasePopulatesMetricsAndUsesSeed() {
    placement::TrainingConfig config = fastBenchmarkConfig();

    const placement::BenchmarkCase test_case{42, 1, 4, 4242};
    const placement::BenchmarkResult first =
        placement::runBenchmarkCase(test_case, config);
    const placement::BenchmarkResult second =
        placement::runBenchmarkCase(test_case, config);

    expect(first.test_id == test_case.test_id, "benchmark result test id");
    expect(first.num_macros == test_case.num_macros, "benchmark result macros");
    expect(
        first.num_std_cells == test_case.num_std_cells,
        "benchmark result standard cells");
    expect(first.seed == test_case.seed, "benchmark result seed");
    expect(first.device == config.device, "benchmark result device");
    expect(first.total_cells == 5, "benchmark result total cells");
    expect(first.num_nets >= 0, "benchmark result net count");
    expect(first.elapsed_seconds >= 0.0, "benchmark result elapsed time");
    expect(first.num_cells_with_overlaps >= 0, "benchmark overlap cell lower bound");
    expect(
        first.num_cells_with_overlaps <= first.total_cells,
        "benchmark overlap cell upper bound");
    expect(
        first.overlap_ratio >= 0.0 && first.overlap_ratio <= 1.0,
        "benchmark overlap ratio range");
    expect(std::isfinite(first.normalized_wl), "benchmark finite wirelength");
    expect(first.normalized_wl >= 0.0, "benchmark nonnegative wirelength");
    expect(
        first.passed == (first.num_cells_with_overlaps == 0),
        "benchmark pass flag");
    expect(first.epochs_completed == config.num_epochs, "benchmark epochs completed");
    expect(first.stopped_early == false, "benchmark early stop flag");
    expect(first.best_epoch == -1, "benchmark best epoch");

    expect(first.num_nets == second.num_nets, "benchmark seeded net count");
    expectNear(
        first.overlap_ratio,
        second.overlap_ratio,
        1e-12,
        "benchmark seeded overlap ratio");
    expectNear(
        first.normalized_wl,
        second.normalized_wl,
        1e-12,
        "benchmark seeded normalized wirelength");
}

void benchmarkSummaryAggregatesOrderedResults() {
    placement::TrainingConfig config = fastBenchmarkConfig();

    const std::vector<placement::BenchmarkCase> cases = {
        {101, 0, 2, 1101},
        {102, 0, 3, 1102},
    };

    const placement::BenchmarkSummary summary =
        placement::runBenchmarkCases(cases, config);

    expect(summary.results.size() == cases.size(), "benchmark summary result count");
    expect(summary.results[0].test_id == 101, "benchmark summary preserves first id");
    expect(summary.results[1].test_id == 102, "benchmark summary preserves second id");

    const double expected_average_overlap =
        (summary.results[0].overlap_ratio + summary.results[1].overlap_ratio) /
        2.0;
    const double expected_average_wirelength =
        (summary.results[0].normalized_wl + summary.results[1].normalized_wl) /
        2.0;
    expectNear(
        summary.average_overlap,
        expected_average_overlap,
        1e-12,
        "benchmark average overlap");
    expectNear(
        summary.average_wirelength,
        expected_average_wirelength,
        1e-12,
        "benchmark average wirelength");

    const int expected_passed =
        (summary.results[0].passed ? 1 : 0) + (summary.results[1].passed ? 1 : 0);
    expect(summary.passed_count == expected_passed, "benchmark passed count");
    expect(
        summary.failed_count ==
            static_cast<int>(summary.results.size()) - expected_passed,
        "benchmark failed count");
    expect(
        std::isfinite(summary.total_elapsed_seconds) &&
            summary.total_elapsed_seconds >= 0.0,
        "benchmark finite total elapsed time");

    const placement::BenchmarkSummary parallel_summary =
        placement::runBenchmarkCases(cases, config, 2);
    expect(
        parallel_summary.results.size() == cases.size(),
        "parallel benchmark summary result count");
    expect(
        parallel_summary.results[0].test_id == summary.results[0].test_id,
        "parallel benchmark preserves first id");
    expect(
        parallel_summary.results[1].test_id == summary.results[1].test_id,
        "parallel benchmark preserves second id");
    expectNear(
        parallel_summary.average_overlap,
        summary.average_overlap,
        1e-12,
        "parallel benchmark average overlap");
    expectNear(
        parallel_summary.average_wirelength,
        summary.average_wirelength,
        1e-12,
        "parallel benchmark average wirelength");
}

void emptyBenchmarkSummaryIsZeroed() {
    const placement::BenchmarkSummary summary = placement::runBenchmarkCases({});

    expect(summary.results.empty(), "empty benchmark result list");
    expectNear(summary.average_overlap, 0.0, 1e-12, "empty average overlap");
    expectNear(summary.average_wirelength, 0.0, 1e-12, "empty average wirelength");
    expectNear(summary.total_elapsed_seconds, 0.0, 1e-12, "empty elapsed time");
    expect(summary.passed_count == 0, "empty passed count");
    expect(summary.failed_count == 0, "empty failed count");
}

}  // namespace

int main() {
    try {
        deterministicMetricsMatchPythonReference();
        generatedProblemCanBeMeasured();
        deterministicLossesMatchPythonReference();
        lossEdgeCasesStayFinite();
        lossesBackpropagateThroughCellPositions();
        trainingWithNoEpochsReturnsInitialPlacement();
        trainingReducesOverlapLoss();
        trainingReducesWirelengthLoss();
        trainingReportsEarlyStopMetadata();
        activeBenchmarkCasesMatchPythonReference();
        benchmarkCasePopulatesMetricsAndUsesSeed();
        benchmarkSummaryAggregatesOrderedResults();
        emptyBenchmarkSummaryIsZeroed();
        visualizationWritesPngWithExpectedContent();
    } catch (const std::exception& error) {
        std::cerr << "placement_unit_tests failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "placement_unit_tests passed\n";
    return 0;
}
