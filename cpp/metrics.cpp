#include "placement/metrics.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

using namespace torch::indexing;

int64_t featureIndex(placement::CellFeatureIdx idx) {
    return static_cast<int64_t>(idx);
}

int64_t featureIndex(placement::PinFeatureIdx idx) {
    return static_cast<int64_t>(idx);
}

int toMetricInt(int64_t value) {
    return static_cast<int>(
        std::min<int64_t>(value, std::numeric_limits<int>::max()));
}

torch::Tensor upperTriangleMask(int64_t size, const torch::Device& device) {
    const auto options = torch::TensorOptions().dtype(torch::kInt64).device(device);
    const auto indices = torch::arange(size, options);
    return indices.unsqueeze(1) < indices.unsqueeze(0);
}

torch::Tensor computePairwiseOverlapAreasForMetrics(
    const torch::Tensor& cell_features) {
    const int64_t num_cells = cell_features.size(0);
    if (num_cells <= 1) {
        return torch::zeros({num_cells, num_cells}, cell_features.options());
    }

    const auto x_col =
        cell_features.index({Slice(), featureIndex(placement::CellFeatureIdx::X)});
    const auto y_col =
        cell_features.index({Slice(), featureIndex(placement::CellFeatureIdx::Y)});
    const auto widths =
        cell_features.index({Slice(), featureIndex(placement::CellFeatureIdx::Width)});
    const auto heights =
        cell_features.index({Slice(), featureIndex(placement::CellFeatureIdx::Height)});

    const auto x_delta = torch::abs(x_col.unsqueeze(1) - x_col.unsqueeze(0));
    const auto y_delta = torch::abs(y_col.unsqueeze(1) - y_col.unsqueeze(0));
    const auto x_span = (widths.unsqueeze(1) + widths.unsqueeze(0)) / 2.0;
    const auto y_span = (heights.unsqueeze(1) + heights.unsqueeze(0)) / 2.0;

    const auto overlap_x = torch::relu(x_span - x_delta);
    const auto overlap_y = torch::relu(y_span - y_delta);
    return overlap_x * overlap_y;
}

double calculateAverageSmoothWirelength(
    const torch::Tensor& cell_features,
    const torch::Tensor& pin_features,
    const torch::Tensor& edge_list) {
    const int64_t num_edges = edge_list.size(0);
    if (num_edges == 0) {
        return 0.0;
    }

    const auto cell_positions = cell_features.index(
        {Slice(),
         Slice(
             featureIndex(placement::CellFeatureIdx::X),
             featureIndex(placement::CellFeatureIdx::Y) + 1)});
    const auto cell_indices =
        pin_features.index({Slice(), featureIndex(placement::PinFeatureIdx::CellIdx)})
            .to(torch::kInt64);
    const auto pin_cell_positions = cell_positions.index_select(0, cell_indices);

    const auto pin_absolute_x =
        pin_cell_positions.index({Slice(), 0}) +
        pin_features.index({Slice(), featureIndex(placement::PinFeatureIdx::PinX)});
    const auto pin_absolute_y =
        pin_cell_positions.index({Slice(), 1}) +
        pin_features.index({Slice(), featureIndex(placement::PinFeatureIdx::PinY)});

    const auto src_pins = edge_list.index({Slice(), 0}).to(torch::kInt64);
    const auto tgt_pins = edge_list.index({Slice(), 1}).to(torch::kInt64);

    const auto dx = torch::abs(
        pin_absolute_x.index_select(0, src_pins) -
        pin_absolute_x.index_select(0, tgt_pins));
    const auto dy = torch::abs(
        pin_absolute_y.index_select(0, src_pins) -
        pin_absolute_y.index_select(0, tgt_pins));

    constexpr double kAlpha = 0.1;
    const auto smooth_manhattan =
        kAlpha * torch::logsumexp(torch::stack({dx / kAlpha, dy / kAlpha}, 0), 0);

    return smooth_manhattan.mean().item<double>();
}

}  // namespace

namespace placement {

OverlapMetrics calculateOverlapMetrics(const torch::Tensor& cell_features) {
    OverlapMetrics metrics;
    const int64_t num_cells = cell_features.size(0);
    if (num_cells <= 1) {
        return metrics;
    }

    const auto pairwise_overlap_area =
        computePairwiseOverlapAreasForMetrics(cell_features);
    const auto mask = upperTriangleMask(num_cells, cell_features.device());
    const auto active_overlap_areas = pairwise_overlap_area.masked_select(mask);
    const auto overlapping_pairs = active_overlap_areas > 0;

    const int64_t overlap_count = overlapping_pairs.sum().item<int64_t>();
    metrics.overlap_count = toMetricInt(overlap_count);
    metrics.total_overlap_area = active_overlap_areas.sum().item<double>();
    metrics.max_overlap_area =
        active_overlap_areas.numel() == 0
            ? 0.0
            : active_overlap_areas.max().item<double>();

    const auto upper_overlap_matrix =
        torch::logical_and(pairwise_overlap_area > 0, mask);
    const auto overlap_matrix =
        torch::logical_or(upper_overlap_matrix, upper_overlap_matrix.transpose(0, 1));
    metrics.cells_with_overlap =
        toMetricInt(overlap_matrix.any(0).sum().item<int64_t>());

    const double total_area =
        cell_features.index({Slice(), featureIndex(CellFeatureIdx::Area)})
            .sum()
            .item<double>();
    metrics.overlap_percentage =
        total_area > 0.0
            ? static_cast<double>(overlap_count) / static_cast<double>(num_cells) *
                  100.0
            : 0.0;
    metrics.has_zero_overlap = metrics.overlap_count == 0;
    return metrics;
}

Metrics calculateNormalizedMetrics(
    const torch::Tensor& cell_features,
    const torch::Tensor& pin_features,
    const torch::Tensor& edge_list) {
    Metrics metrics;
    metrics.total_cells = cell_features.size(0);
    metrics.num_nets = edge_list.size(0);

    const OverlapMetrics overlap_metrics = calculateOverlapMetrics(cell_features);
    metrics.num_cells_with_overlaps = overlap_metrics.cells_with_overlap;
    metrics.overlap_ratio =
        metrics.total_cells > 0
            ? static_cast<double>(metrics.num_cells_with_overlaps) /
                  static_cast<double>(metrics.total_cells)
            : 0.0;

    if (metrics.num_nets == 0) {
        return metrics;
    }

    const double total_area =
        cell_features.index({Slice(), featureIndex(CellFeatureIdx::Area)})
            .sum()
            .item<double>();
    if (total_area <= 0.0) {
        return metrics;
    }

    metrics.normalized_wl =
        calculateAverageSmoothWirelength(cell_features, pin_features, edge_list) /
        std::sqrt(total_area);
    return metrics;
}

}  // namespace placement
