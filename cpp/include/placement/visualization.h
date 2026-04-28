#pragma once

#include <torch/torch.h>

#include <filesystem>

namespace placement {

void plotPlacement(
    const torch::Tensor& initial_cell_features,
    const torch::Tensor& final_cell_features,
    const std::filesystem::path& output_path);

}  // namespace placement
