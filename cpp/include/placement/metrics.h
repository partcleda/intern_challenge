#pragma once

#include "placement/types.h"

namespace placement {

OverlapMetrics calculateOverlapMetrics(const torch::Tensor& cell_features);

Metrics calculateNormalizedMetrics(
    const torch::Tensor& cell_features,
    const torch::Tensor& pin_features,
    const torch::Tensor& edge_list);

}  // namespace placement
