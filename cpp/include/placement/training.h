#pragma once

#include "placement/types.h"

namespace placement {

TrainingResult trainPlacement(
    const torch::Tensor& cell_features,
    const torch::Tensor& pin_features,
    const torch::Tensor& edge_list,
    const TrainingConfig& config = {});

}  // namespace placement
