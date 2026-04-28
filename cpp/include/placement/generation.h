#pragma once

#include "placement/types.h"

namespace placement {

PlacementProblem generatePlacementInput(
    int num_macros,
    int num_std_cells,
    const torch::Device& device = torch::kCPU,
    bool verbose = true);

void initializeCellPositions(torch::Tensor& cell_features, double spread_scale = 0.6);

}  // namespace placement
