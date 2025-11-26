#include <torch/extension.h>

// Forward/backward declarations implemented in overlap_cuda_kernel.cu

torch::Tensor overlap_forward_cuda(
    torch::Tensor positions,
    torch::Tensor widths,
    torch::Tensor heights,
    torch::Tensor areas,
    torch::Tensor cell_bins,
    torch::Tensor bin_starts,
    torch::Tensor bin_counts,
    torch::Tensor sorted_indices,
    int64_t num_bins_x,
    int64_t num_bins_y,
    torch::Tensor pair_bins_a,
    torch::Tensor pair_bins_b,
    double margin_factor,
    double alpha,
    double scale);

torch::Tensor overlap_backward_cuda(
    torch::Tensor positions,
    torch::Tensor widths,
    torch::Tensor heights,
    torch::Tensor areas,
    torch::Tensor cell_bins,
    torch::Tensor bin_starts,
    torch::Tensor bin_counts,
    torch::Tensor sorted_indices,
    int64_t num_bins_x,
    int64_t num_bins_y,
    torch::Tensor pair_bins_a,
    torch::Tensor pair_bins_b,
    double margin_factor,
    double alpha,
    double scale);


torch::Tensor overlap_forward(
    torch::Tensor positions,
    torch::Tensor widths,
    torch::Tensor heights,
    torch::Tensor areas,
    torch::Tensor cell_bins,
    torch::Tensor bin_starts,
    torch::Tensor bin_counts,
    torch::Tensor sorted_indices,
    int64_t num_bins_x,
    int64_t num_bins_y,
    torch::Tensor pair_bins_a,
    torch::Tensor pair_bins_b,
    double margin_factor,
    double alpha,
    double scale) {
  TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
  return overlap_forward_cuda(
      positions,
      widths,
      heights,
      areas,
      cell_bins,
      bin_starts,
      bin_counts,
      sorted_indices,
      num_bins_x,
      num_bins_y,
      pair_bins_a,
      pair_bins_b,
      margin_factor,
      alpha,
      scale);
}


torch::Tensor overlap_backward(
    torch::Tensor positions,
    torch::Tensor widths,
    torch::Tensor heights,
    torch::Tensor areas,
    torch::Tensor cell_bins,
    torch::Tensor bin_starts,
    torch::Tensor bin_counts,
    torch::Tensor sorted_indices,
    int64_t num_bins_x,
    int64_t num_bins_y,
    torch::Tensor pair_bins_a,
    torch::Tensor pair_bins_b,
    double margin_factor,
    double alpha,
    double scale) {
  TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
  return overlap_backward_cuda(
      positions,
      widths,
      heights,
      areas,
      cell_bins,
      bin_starts,
      bin_counts,
      sorted_indices,
      num_bins_x,
      num_bins_y,
      pair_bins_a,
      pair_bins_b,
      margin_factor,
      alpha,
      scale);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &overlap_forward, "Overlap loss forward (CUDA)");
  m.def("backward", &overlap_backward, "Overlap loss backward (CUDA)");
}
