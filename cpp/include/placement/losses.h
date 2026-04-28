#pragma once

#include "placement/types.h"

namespace placement {

torch::Tensor computePairwiseOverlapAreas(const torch::Tensor& cell_features);

torch::Tensor wirelengthAttractionLoss(
    const torch::Tensor& cell_features,
    const torch::Tensor& pin_features,
    const torch::Tensor& edge_list);

torch::Tensor overlapRepulsionLoss(
    const torch::Tensor& cell_features,
    const torch::Tensor& pin_features,
    const torch::Tensor& edge_list);

}  // namespace placement
