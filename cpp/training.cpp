#include "placement/training.h"

#include "placement/losses.h"
#include "placement/metrics.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

using namespace torch::indexing;

int64_t featureIndex(placement::CellFeatureIdx idx) {
    return static_cast<int64_t>(idx);
}

torch::Tensor makeCellFeaturesWithPositions(
    const torch::Tensor& cell_features,
    const torch::Tensor& cell_positions) {
    const auto prefix = cell_features.index(
        {Slice(), Slice(0, featureIndex(placement::CellFeatureIdx::X))});
    const auto suffix = cell_features.index(
        {Slice(), Slice(featureIndex(placement::CellFeatureIdx::Width), None)});
    return torch::cat({prefix, cell_positions, suffix}, 1);
}

double optimizerLearningRate(torch::optim::Adam& optimizer) {
    return static_cast<torch::optim::AdamOptions&>(
               optimizer.param_groups().front().options())
        .lr();
}

void setOptimizerLearningRate(torch::optim::Adam& optimizer, double lr) {
    for (auto& group : optimizer.param_groups()) {
        static_cast<torch::optim::AdamOptions&>(group.options()).lr(lr);
    }
}

void clipGradientNorm(const torch::Tensor& tensor, double max_norm) {
    if (!tensor.grad().defined()) {
        return;
    }

    const double grad_norm = tensor.grad().norm().item<double>();
    if (std::isfinite(grad_norm) && grad_norm > max_norm) {
        const double scale = max_norm / (grad_norm + 1e-12);
        tensor.grad().mul_(scale);
    }
}

std::tm localTime(std::time_t time) {
    std::tm local_time{};
#if defined(_WIN32)
    localtime_s(&local_time, &time);
#else
    localtime_r(&time, &local_time);
#endif
    return local_time;
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

class LearningRateScheduler {
public:
    LearningRateScheduler(const placement::TrainingConfig& config, double base_lr)
        : config_(config), base_lr_(base_lr), current_lr_(base_lr) {
        if (config_.scheduler_name != "none" &&
            config_.scheduler_name != "plateau" &&
            config_.scheduler_name != "cosine" &&
            config_.scheduler_name != "step" &&
            config_.scheduler_name != "exponential") {
            throw std::invalid_argument(
                "Unsupported scheduler: " + config_.scheduler_name);
        }
    }

    void step(torch::optim::Adam& optimizer, double metric) {
        if (config_.scheduler_name == "none") {
            return;
        }

        ++epoch_;
        if (config_.scheduler_name == "plateau") {
            stepPlateau(metric);
        } else if (config_.scheduler_name == "cosine") {
            stepCosine();
        } else if (config_.scheduler_name == "step") {
            stepStep();
        } else if (config_.scheduler_name == "exponential") {
            current_lr_ *= config_.scheduler_gamma;
        }

        setOptimizerLearningRate(optimizer, current_lr_);
    }

private:
    void stepPlateau(double metric) {
        if (metric < best_metric_) {
            best_metric_ = metric;
            bad_epochs_ = 0;
            return;
        }

        ++bad_epochs_;
        if (bad_epochs_ > config_.scheduler_patience) {
            current_lr_ *= config_.scheduler_factor;
            bad_epochs_ = 0;
        }
    }

    void stepCosine() {
        const int t_max = std::max(1, config_.num_epochs);
        const double progress =
            static_cast<double>(std::min(epoch_, t_max)) / static_cast<double>(t_max);
        current_lr_ =
            config_.scheduler_eta_min +
            (base_lr_ - config_.scheduler_eta_min) *
                (1.0 + std::cos(std::acos(-1.0) * progress)) / 2.0;
    }

    void stepStep() {
        const int step_size = std::max(1, config_.scheduler_step_size);
        if (epoch_ % step_size == 0) {
            current_lr_ *= config_.scheduler_gamma;
        }
    }

    const placement::TrainingConfig& config_;
    double base_lr_ = 0.0;
    double current_lr_ = 0.0;
    double best_metric_ = std::numeric_limits<double>::infinity();
    int bad_epochs_ = 0;
    int epoch_ = 0;
};

}  // namespace

namespace placement {

TrainingResult trainPlacement(
    const torch::Tensor& cell_features,
    const torch::Tensor& pin_features,
    const torch::Tensor& edge_list,
    const TrainingConfig& config) {
    TrainingResult result;
    result.run_started_at =
        isoTimestampSeconds(std::chrono::system_clock::now());
    auto working_cell_features = cell_features.clone();
    auto working_pin_features = pin_features.to(working_cell_features.device());
    auto working_edge_list = edge_list.to(working_cell_features.device());

    result.initial_cell_features = working_cell_features.clone();
    if (config.num_epochs <= 0 || working_cell_features.size(0) == 0) {
        result.final_cell_features = working_cell_features.clone();
        return result;
    }

    auto cell_positions =
        working_cell_features
            .index({Slice(),
                    Slice(
                        featureIndex(CellFeatureIdx::X),
                        featureIndex(CellFeatureIdx::Y) + 1)})
            .clone()
            .detach();
    cell_positions.set_requires_grad(true);

    torch::optim::Adam optimizer(
        {cell_positions},
        torch::optim::AdamOptions(config.lr));
    LearningRateScheduler scheduler(config, config.lr);

    auto best_cell_positions = cell_positions.detach().clone();
    double best_overlap_score = std::numeric_limits<double>::infinity();
    double best_zero_overlap_wl = std::numeric_limits<double>::infinity();
    int epochs_without_improvement = 0;
    int zero_overlap_epochs_without_improvement = 0;
    bool zero_overlap_reached = false;

    for (int epoch = 0; epoch < config.num_epochs; ++epoch) {
        result.epochs_completed = epoch + 1;
        optimizer.zero_grad();

        auto current_cell_features =
            makeCellFeaturesWithPositions(working_cell_features, cell_positions);
        const auto wl_loss = wirelengthAttractionLoss(
            current_cell_features,
            working_pin_features,
            working_edge_list);
        const auto overlap_loss = overlapRepulsionLoss(
            current_cell_features,
            working_pin_features,
            working_edge_list);
        const auto total_loss =
            config.lambda_wirelength * wl_loss +
            config.lambda_overlap * overlap_loss;

        total_loss.backward();
        clipGradientNorm(cell_positions, 5.0);
        optimizer.step();
        scheduler.step(optimizer, total_loss.item<double>());

        const bool should_log_epoch =
            config.verbose &&
            ((config.log_interval > 0 && epoch % config.log_interval == 0) ||
             epoch == config.num_epochs - 1);
        const bool should_compute_overlap_metrics =
            config.track_overlap_metrics ||
            config.early_stop_enabled ||
            should_log_epoch;

        OverlapMetrics overlap_metrics;
        torch::Tensor updated_cell_features;
        if (should_compute_overlap_metrics) {
            updated_cell_features = makeCellFeaturesWithPositions(
                working_cell_features,
                cell_positions.detach());
            overlap_metrics = calculateOverlapMetrics(updated_cell_features);
        }

        if (config.early_stop_enabled) {
            const double overlap_score = overlap_metrics.total_overlap_area;
            const bool has_zero_overlap =
                overlap_metrics.overlap_count == 0 ||
                overlap_score <= config.early_stop_overlap_threshold;

            if (has_zero_overlap) {
                const double current_wl = wirelengthAttractionLoss(
                                              updated_cell_features,
                                              working_pin_features,
                                              working_edge_list)
                                              .item<double>();
                if (!zero_overlap_reached ||
                    current_wl <
                        best_zero_overlap_wl - config.early_stop_min_delta) {
                    zero_overlap_reached = true;
                    best_zero_overlap_wl = current_wl;
                    best_cell_positions = cell_positions.detach().clone();
                    result.best_epoch = epoch;
                    zero_overlap_epochs_without_improvement = 0;
                } else {
                    ++zero_overlap_epochs_without_improvement;
                }

                if (zero_overlap_reached &&
                    zero_overlap_epochs_without_improvement >=
                        config.early_stop_zero_overlap_patience) {
                    result.stopped_early = true;
                    result.stop_reason = "zero_overlap_plateau";
                }
            } else {
                if (zero_overlap_reached) {
                    ++zero_overlap_epochs_without_improvement;
                    if (zero_overlap_epochs_without_improvement >=
                        config.early_stop_zero_overlap_patience) {
                        result.stopped_early = true;
                        result.stop_reason = "zero_overlap_plateau";
                    }
                } else {
                    if (overlap_score <
                        best_overlap_score - config.early_stop_min_delta) {
                        best_overlap_score = overlap_score;
                        best_cell_positions = cell_positions.detach().clone();
                        result.best_epoch = epoch;
                        epochs_without_improvement = 0;
                    } else {
                        ++epochs_without_improvement;
                    }

                    if (epochs_without_improvement >=
                        config.early_stop_patience) {
                        result.stopped_early = true;
                        result.stop_reason = "overlap_plateau";
                    }
                }
            }
        }

        if (config.track_loss_history) {
            result.loss_history.total_loss.push_back(total_loss.item<double>());
            result.loss_history.wirelength_loss.push_back(wl_loss.item<double>());
            result.loss_history.overlap_loss.push_back(overlap_loss.item<double>());
            result.loss_history.learning_rate.push_back(
                optimizerLearningRate(optimizer));
            if (config.track_overlap_metrics) {
                result.loss_history.overlap_count.push_back(
                    overlap_metrics.overlap_count);
                result.loss_history.total_overlap_area.push_back(
                    overlap_metrics.total_overlap_area);
                result.loss_history.max_overlap_area.push_back(
                    overlap_metrics.max_overlap_area);
            }
        }

        if (should_log_epoch) {
            std::cout << "Epoch " << epoch << "/" << config.num_epochs << ":\n";
            std::cout << "  Total Loss: " << total_loss.item<double>() << "\n";
            std::cout << "  Wirelength Loss: " << wl_loss.item<double>() << "\n";
            std::cout << "  Overlap Loss: " << overlap_loss.item<double>() << "\n";
            std::cout << "  Learning Rate: " << optimizerLearningRate(optimizer)
                      << "\n";
            if (should_compute_overlap_metrics) {
                std::cout << "  Overlap Count: "
                          << overlap_metrics.overlap_count << "\n";
                std::cout << "  Total Overlap Area: "
                          << overlap_metrics.total_overlap_area << "\n";
            }
            if (config.early_stop_enabled) {
                std::cout << "  Best Epoch: " << result.best_epoch << "\n";
            }
        }

        if (result.stopped_early) {
            if (config.verbose) {
                std::cout << "Early stopping at epoch " << epoch
                          << " with reason=" << result.stop_reason
                          << " best_epoch=" << result.best_epoch << "\n";
            }
            break;
        }
    }

    const auto final_positions =
        config.early_stop_enabled ? best_cell_positions : cell_positions.detach();
    result.final_cell_features =
        makeCellFeaturesWithPositions(working_cell_features, final_positions);
    return result;
}

}  // namespace placement
