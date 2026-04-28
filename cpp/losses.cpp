#include "placement/losses.h"

#include <cstdint>

namespace {

using namespace torch::indexing;

int64_t featureIndex(placement::CellFeatureIdx idx) {
    return static_cast<int64_t>(idx);
}

int64_t featureIndex(placement::PinFeatureIdx idx) {
    return static_cast<int64_t>(idx);
}

torch::Tensor differentiableZero(const torch::Tensor& like) {
    auto zero = torch::zeros({}, like.options());
    zero.set_requires_grad(true);
    return zero;
}

}  // namespace

namespace placement {

torch::Tensor computePairwiseOverlapAreas(const torch::Tensor& cell_features) {
    const int64_t num_cells = cell_features.size(0);
    if (num_cells <= 1) {
        return torch::zeros({num_cells, num_cells}, cell_features.options());
    }

    const auto x_col =
        cell_features.index({Slice(), featureIndex(CellFeatureIdx::X)});
    const auto y_col =
        cell_features.index({Slice(), featureIndex(CellFeatureIdx::Y)});
    const auto widths =
        cell_features.index({Slice(), featureIndex(CellFeatureIdx::Width)});
    const auto heights =
        cell_features.index({Slice(), featureIndex(CellFeatureIdx::Height)});

    const auto x_delta = torch::abs(x_col.unsqueeze(1) - x_col.unsqueeze(0));
    const auto y_delta = torch::abs(y_col.unsqueeze(1) - y_col.unsqueeze(0));
    const auto x_span = (widths.unsqueeze(1) + widths.unsqueeze(0)) / 2.0;
    const auto y_span = (heights.unsqueeze(1) + heights.unsqueeze(0)) / 2.0;

    const auto overlap_x = torch::relu(x_span - x_delta);
    const auto overlap_y = torch::relu(y_span - y_delta);
    return overlap_x * overlap_y;
}

torch::Tensor wirelengthAttractionLoss(
    const torch::Tensor& cell_features,
    const torch::Tensor& pin_features,
    const torch::Tensor& edge_list) {
    const int64_t num_edges = edge_list.size(0);
    if (num_edges == 0) {
        return differentiableZero(cell_features);
    }

    const auto cell_positions = cell_features.index(
        {Slice(),
         Slice(
             featureIndex(CellFeatureIdx::X),
             featureIndex(CellFeatureIdx::Y) + 1)});
    const auto cell_indices =
        pin_features.index({Slice(), featureIndex(PinFeatureIdx::CellIdx)})
            .to(torch::kInt64);

    const auto pin_cell_positions = cell_positions.index_select(0, cell_indices);
    const auto pin_absolute_x =
        pin_cell_positions.index({Slice(), 0}) +
        pin_features.index({Slice(), featureIndex(PinFeatureIdx::PinX)});
    const auto pin_absolute_y =
        pin_cell_positions.index({Slice(), 1}) +
        pin_features.index({Slice(), featureIndex(PinFeatureIdx::PinY)});

    const auto src_pins = edge_list.index({Slice(), 0}).to(torch::kInt64);
    const auto tgt_pins = edge_list.index({Slice(), 1}).to(torch::kInt64);

    const auto src_x = pin_absolute_x.index_select(0, src_pins);
    const auto src_y = pin_absolute_y.index_select(0, src_pins);
    const auto tgt_x = pin_absolute_x.index_select(0, tgt_pins);
    const auto tgt_y = pin_absolute_y.index_select(0, tgt_pins);

    constexpr double kAlpha = 0.1;
    const auto dx = torch::abs(src_x - tgt_x);
    const auto dy = torch::abs(src_y - tgt_y);
    const auto smooth_manhattan =
        kAlpha * torch::logsumexp(torch::stack({dx / kAlpha, dy / kAlpha}, 0), 0);

    return smooth_manhattan.sum() / static_cast<double>(num_edges);
}

torch::Tensor overlapRepulsionLoss(
    const torch::Tensor& cell_features,
    const torch::Tensor& pin_features,
    const torch::Tensor& edge_list) {
    (void)pin_features;
    (void)edge_list;

    const int64_t num_cells = cell_features.size(0);
    if (num_cells <= 1) {
        return differentiableZero(cell_features);
    }

    const auto pairwise_overlap_area = computePairwiseOverlapAreas(cell_features);
    const auto mask = torch::triu(torch::ones_like(pairwise_overlap_area), 1);

    constexpr double kOverlapScalar = 200.0;
    return torch::log1p(torch::sum(pairwise_overlap_area * mask)) * kOverlapScalar;
}

}  // namespace placement
