#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

namespace placement {

enum class CellFeatureIdx : int64_t {
    Area = 0,
    NumPins = 1,
    X = 2,
    Y = 3,
    Width = 4,
    Height = 5,
};

enum class PinFeatureIdx : int64_t {
    CellIdx = 0,
    PinX = 1,
    PinY = 2,
    X = 3,
    Y = 4,
    Width = 5,
    Height = 6,
};

struct PlacementProblem {
    torch::Tensor cell_features;
    torch::Tensor pin_features;
    torch::Tensor edge_list;
};

struct TrainingConfig {
    c10::DeviceType device = torch::kCPU;
    int num_epochs = 1000;
    double lr = 0.1;
    double lambda_wirelength = 3.0;
    double lambda_overlap = 1.0;
    std::string scheduler_name = "plateau";
    int scheduler_patience = 50;
    double scheduler_factor = 0.5;
    double scheduler_eta_min = 1e-4;
    int scheduler_step_size = 100;
    double scheduler_gamma = 0.95;
    bool track_loss_history = false;
    bool track_overlap_metrics = false;
    bool early_stop_enabled = true;
    int early_stop_patience = 75;
    double early_stop_min_delta = 1e-4;
    double early_stop_overlap_threshold = 1e-4;
    int early_stop_zero_overlap_patience = 25;
    bool verbose = true;
    int log_interval = 100;
};

struct LossHistory {
    std::vector<double> total_loss;
    std::vector<double> wirelength_loss;
    std::vector<double> overlap_loss;
    std::vector<double> learning_rate;
    std::vector<int> overlap_count;
    std::vector<double> total_overlap_area;
    std::vector<double> max_overlap_area;
};

struct TrainingResult {
    torch::Tensor final_cell_features;
    torch::Tensor initial_cell_features;
    bool stopped_early = false;
    std::string stop_reason;
    std::string run_started_at;
    int best_epoch = -1;
    int epochs_completed = 0;
    LossHistory loss_history;
};

struct OverlapMetrics {
    int overlap_count = 0;
    double total_overlap_area = 0.0;
    double max_overlap_area = 0.0;
    double overlap_percentage = 0.0;
    int cells_with_overlap = 0;
    bool has_zero_overlap = true;
};

struct Metrics {
    double overlap_ratio = 0.0;
    double normalized_wl = 0.0;
    int num_cells_with_overlaps = 0;
    int64_t total_cells = 0;
    int64_t num_nets = 0;
};

struct BenchmarkCase {
    int test_id = 0;
    int num_macros = 0;
    int num_std_cells = 0;
    int seed = 0;
};

struct BenchmarkResult {
    int test_id = 0;
    int num_macros = 0;
    int num_std_cells = 0;
    int64_t total_cells = 0;
    int64_t total_pins = 0;
    int64_t num_nets = 0;
    int seed = 0;
    c10::DeviceType device = torch::kCPU;
    double elapsed_seconds = 0.0;
    int num_cells_with_overlaps = 0;
    double overlap_ratio = 0.0;
    double normalized_wl = 0.0;
    bool passed = false;
    bool stopped_early = false;
    std::string stop_reason;
    std::string run_started_at;
    int best_epoch = -1;
    int epochs_completed = 0;
    LossHistory loss_history;
};

struct BenchmarkSummary {
    std::vector<BenchmarkResult> results;
    double average_overlap = 0.0;
    double average_wirelength = 0.0;
    double total_elapsed_seconds = 0.0;
    int passed_count = 0;
    int failed_count = 0;
};

}  // namespace placement
