#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace {
constexpr float kEps = 1e-8f;
constexpr float kAreaEps = 1e-12f;
constexpr int TILE_SIZE = 64;
constexpr int BLOCK_THREADS = 128;

__device__ inline float clampf(float v, float lo, float hi) {
  return fminf(fmaxf(v, lo), hi);
}

__device__ inline float softplus_scaled(float input, float alpha) {
  float alpha_safe = fmaxf(alpha, 1e-8f);
  float clamped = clampf(input, -50.0f, 50.0f);
  float scaled = clampf(clamped * alpha_safe, -50.0f, 50.0f);
  float val;
  if (scaled > 20.0f) {
    val = scaled;
  } else if (scaled < -20.0f) {
    val = expf(scaled);
  } else {
    val = log1pf(expf(scaled));
  }
  val = val / alpha_safe;
  return fmaxf(val, 0.0f);
}

__device__ inline float strong_sigmoid_grad(float input, float alpha, float scale) {
  float alpha_safe = fmaxf(alpha, 1e-8f);
  float clamped = clampf(input, -50.0f, 50.0f);
  float scaled = clampf(clamped * alpha_safe, -50.0f, 50.0f);
  float sigmoid_val = 1.0f / (1.0f + expf(-scaled));
  float tanh_arg = clampf(input * 0.5f, -10.0f, 10.0f);
  float magnitude = 1.0f + 2.0f * tanhf(tanh_arg);
  return scale * sigmoid_val * magnitude;
}

__device__ inline float sign_grad(float value) {
  if (value > 0.0f) {
    return 1.0f;
  } else if (value < 0.0f) {
    return -1.0f;
  }
  return 0.0f;
}

__device__ inline float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ inline void block_reduce_add(float val, float* out) {
  __shared__ float warp_sums[BLOCK_THREADS / 32];
  float sum = warp_reduce_sum(val);
  if ((threadIdx.x & 31) == 0) {
    warp_sums[threadIdx.x >> 5] = sum;
  }
  __syncthreads();
  if (threadIdx.x < BLOCK_THREADS / 32) {
    float block_sum = warp_sums[threadIdx.x];
    block_sum = warp_reduce_sum(block_sum);
    if (threadIdx.x == 0) {
      atomicAdd(out, block_sum);
    }
  }
  __syncthreads();
}

}  // namespace

__global__ void overlap_forward_kernel(
    const float* __restrict__ positions,
    const float* __restrict__ widths,
    const float* __restrict__ heights,
    const float* __restrict__ areas,
    const int32_t* __restrict__ cell_bins,
    const int32_t* __restrict__ bin_starts,
    const int32_t* __restrict__ bin_counts,
    const int32_t* __restrict__ sorted_indices,
    const int32_t* __restrict__ pair_bins_a,
    const int32_t* __restrict__ pair_bins_b,
    int pair_count,
    float margin_factor,
    float alpha,
    float scale,
    float* __restrict__ loss_out) {
  int pair_idx = blockIdx.x;
  if (pair_idx >= pair_count) {
    return;
  }

  int bin_a = pair_bins_a[pair_idx];
  int bin_b = pair_bins_b[pair_idx];
  int start_a = bin_starts[bin_a];
  int count_a = bin_counts[bin_a];
  int start_b = bin_starts[bin_b];
  int count_b = bin_counts[bin_b];
  if (count_a == 0 || count_b == 0) {
    return;
  }

  bool same_bin = bin_a == bin_b;

  __shared__ float sh_pos_x[TILE_SIZE];
  __shared__ float sh_pos_y[TILE_SIZE];
  __shared__ float sh_w[TILE_SIZE];
  __shared__ float sh_h[TILE_SIZE];
  __shared__ float sh_area[TILE_SIZE];

  float thread_loss = 0.0f;

  for (int i_local = threadIdx.x; i_local < count_a; i_local += blockDim.x) {
    int cell_i = sorted_indices[start_a + i_local];
    float xi = positions[cell_i * 2];
    float yi = positions[cell_i * 2 + 1];
    float wi = widths[cell_i];
    float hi = heights[cell_i];
    float areai = areas[cell_i];

    for (int j_tile = 0; j_tile < count_b; j_tile += TILE_SIZE) {
      if (threadIdx.x < TILE_SIZE) {
        int load_idx = j_tile + threadIdx.x;
        if (load_idx < count_b) {
          int cell_j = sorted_indices[start_b + load_idx];
          sh_pos_x[threadIdx.x] = positions[cell_j * 2];
          sh_pos_y[threadIdx.x] = positions[cell_j * 2 + 1];
          sh_w[threadIdx.x] = widths[cell_j];
          sh_h[threadIdx.x] = heights[cell_j];
          sh_area[threadIdx.x] = areas[cell_j];
        }
      }
      __syncthreads();

      int tile_count = min(TILE_SIZE, count_b - j_tile);
      for (int j_local = 0; j_local < tile_count; ++j_local) {
        int global_j = j_tile + j_local;
        if (same_bin && global_j <= i_local) {
          continue;
        }
        float xj = sh_pos_x[j_local];
        float yj = sh_pos_y[j_local];
        float wj = sh_w[j_local];
        float hj = sh_h[j_local];
        float areaj = sh_area[j_local];

        float dx_val = xi - xj;
        float dy_val = yi - yj;
        float abs_dx = fabsf(dx_val);
        float abs_dy = fabsf(dy_val);

        float min_sep_x = 0.5f * (wi + wj);
        float min_sep_y = 0.5f * (hi + hj);
        float min_dim_i = fminf(wi, hi);
        float min_dim_j = fminf(wj, hj);
        float min_dim_pair = fminf(min_dim_i, min_dim_j);
        float margin = margin_factor * min_dim_pair;

        float required_sep_x = min_sep_x + margin;
        float required_sep_y = min_sep_y + margin;

        float overlap_x_raw = required_sep_x - abs_dx;
        float overlap_y_raw = required_sep_y - abs_dy;

        float overlap_x = softplus_scaled(overlap_x_raw, alpha);
        float overlap_y = softplus_scaled(overlap_y_raw, alpha);
        float raw_area = overlap_x * overlap_y;
        if (raw_area <= kEps) {
          continue;
        }

        float overlap_area = fmaxf(raw_area, kEps);
        float repulsion = sqrtf(fmaxf(areai * areaj, kAreaEps));
        float penalty = repulsion * powf(overlap_area, 2.5f);
        if (isfinite(penalty)) {
          thread_loss += penalty;
        }
      }
      __syncthreads();
    }
  }

  block_reduce_add(thread_loss, loss_out);
}

__global__ void overlap_backward_kernel(
    const float* __restrict__ positions,
    const float* __restrict__ widths,
    const float* __restrict__ heights,
    const float* __restrict__ areas,
    const int32_t* __restrict__ cell_bins,
    const int32_t* __restrict__ bin_starts,
    const int32_t* __restrict__ bin_counts,
    const int32_t* __restrict__ sorted_indices,
    const int32_t* __restrict__ pair_bins_a,
    const int32_t* __restrict__ pair_bins_b,
    int pair_count,
    float margin_factor,
    float alpha,
    float scale,
    float* __restrict__ grad_positions) {
  int pair_idx = blockIdx.x;
  if (pair_idx >= pair_count) {
    return;
  }

  int bin_a = pair_bins_a[pair_idx];
  int bin_b = pair_bins_b[pair_idx];
  int start_a = bin_starts[bin_a];
  int count_a = bin_counts[bin_a];
  int start_b = bin_starts[bin_b];
  int count_b = bin_counts[bin_b];
  if (count_a == 0 || count_b == 0) {
    return;
  }

  bool same_bin = bin_a == bin_b;

  __shared__ float sh_pos_x[TILE_SIZE];
  __shared__ float sh_pos_y[TILE_SIZE];
  __shared__ float sh_w[TILE_SIZE];
  __shared__ float sh_h[TILE_SIZE];
  __shared__ float sh_area[TILE_SIZE];

  for (int i_local = threadIdx.x; i_local < count_a; i_local += blockDim.x) {
    int cell_i = sorted_indices[start_a + i_local];
    float xi = positions[cell_i * 2];
    float yi = positions[cell_i * 2 + 1];
    float wi = widths[cell_i];
    float hi = heights[cell_i];
    float areai = areas[cell_i];

    for (int j_tile = 0; j_tile < count_b; j_tile += TILE_SIZE) {
      if (threadIdx.x < TILE_SIZE) {
        int load_idx = j_tile + threadIdx.x;
        if (load_idx < count_b) {
          int cell_j = sorted_indices[start_b + load_idx];
          sh_pos_x[threadIdx.x] = positions[cell_j * 2];
          sh_pos_y[threadIdx.x] = positions[cell_j * 2 + 1];
          sh_w[threadIdx.x] = widths[cell_j];
          sh_h[threadIdx.x] = heights[cell_j];
          sh_area[threadIdx.x] = areas[cell_j];
        }
      }
      __syncthreads();

      int tile_count = min(TILE_SIZE, count_b - j_tile);
      for (int j_local = 0; j_local < tile_count; ++j_local) {
        int global_j = j_tile + j_local;
        if (same_bin && global_j <= i_local) {
          continue;
        }
        int cell_j = sorted_indices[start_b + global_j];
        float xj = sh_pos_x[j_local];
        float yj = sh_pos_y[j_local];
        float wj = sh_w[j_local];
        float hj = sh_h[j_local];
        float areaj = sh_area[j_local];

        float dx_val = xi - xj;
        float dy_val = yi - yj;
        float abs_dx = fabsf(dx_val);
        float abs_dy = fabsf(dy_val);

        float min_sep_x = 0.5f * (wi + wj);
        float min_sep_y = 0.5f * (hi + hj);
        float min_dim_i = fminf(wi, hi);
        float min_dim_j = fminf(wj, hj);
        float min_dim_pair = fminf(min_dim_i, min_dim_j);
        float margin = margin_factor * min_dim_pair;

        float required_sep_x = min_sep_x + margin;
        float required_sep_y = min_sep_y + margin;

        float overlap_x_raw = required_sep_x - abs_dx;
        float overlap_y_raw = required_sep_y - abs_dy;

        float overlap_x = softplus_scaled(overlap_x_raw, alpha);
        float overlap_y = softplus_scaled(overlap_y_raw, alpha);
        float raw_area = overlap_x * overlap_y;
        if (raw_area <= kEps) {
          continue;
        }

        float overlap_area = fmaxf(raw_area, kEps);
        float repulsion = sqrtf(fmaxf(areai * areaj, kAreaEps));
        float pow_term = powf(overlap_area, 1.5f);
        float coeff = repulsion * 2.5f * pow_term;

        float dpenalty_dox = coeff * overlap_y;
        float dpenalty_doy = coeff * overlap_x;

        float grad_factor_x = strong_sigmoid_grad(overlap_x_raw, alpha, scale);
        float grad_factor_y = strong_sigmoid_grad(overlap_y_raw, alpha, scale);

        float sign_dx = sign_grad(dx_val);
        float sign_dy = sign_grad(dy_val);

        float grad_x_i = -dpenalty_dox * grad_factor_x * sign_dx;
        float grad_y_i = -dpenalty_doy * grad_factor_y * sign_dy;
        float grad_x_j = -grad_x_i;
        float grad_y_j = -grad_y_i;

        if (grad_x_i != 0.0f) {
          atomicAdd(&grad_positions[cell_i * 2], grad_x_i);
          atomicAdd(&grad_positions[cell_j * 2], grad_x_j);
        }
        if (grad_y_i != 0.0f) {
          atomicAdd(&grad_positions[cell_i * 2 + 1], grad_y_i);
          atomicAdd(&grad_positions[cell_j * 2 + 1], grad_y_j);
        }
      }
      __syncthreads();
    }
  }
}


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
    double scale) {
  auto loss = torch::zeros({1}, positions.options().dtype(torch::kFloat32));
  int pair_count = pair_bins_a.size(0);
  if (pair_count == 0) {
    return loss.squeeze();
  }

  const dim3 blocks(pair_count);
  const dim3 threads(BLOCK_THREADS);

  overlap_forward_kernel<<<blocks, threads>>>(
      positions.data_ptr<float>(),
      widths.data_ptr<float>(),
      heights.data_ptr<float>(),
      areas.data_ptr<float>(),
      cell_bins.data_ptr<int32_t>(),
      bin_starts.data_ptr<int32_t>(),
      bin_counts.data_ptr<int32_t>(),
      sorted_indices.data_ptr<int32_t>(),
      pair_bins_a.data_ptr<int32_t>(),
      pair_bins_b.data_ptr<int32_t>(),
      pair_count,
      static_cast<float>(margin_factor),
      static_cast<float>(alpha),
      static_cast<float>(scale),
      loss.data_ptr<float>());
  AT_CUDA_CHECK(cudaGetLastError());
  return loss.squeeze();
}


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
    double scale) {
  auto grad = torch::zeros_like(positions);
  int pair_count = pair_bins_a.size(0);
  if (pair_count == 0) {
    return grad;
  }

  const dim3 blocks(pair_count);
  const dim3 threads(BLOCK_THREADS);

  overlap_backward_kernel<<<blocks, threads>>>(
      positions.data_ptr<float>(),
      widths.data_ptr<float>(),
      heights.data_ptr<float>(),
      areas.data_ptr<float>(),
      cell_bins.data_ptr<int32_t>(),
      bin_starts.data_ptr<int32_t>(),
      bin_counts.data_ptr<int32_t>(),
      sorted_indices.data_ptr<int32_t>(),
      pair_bins_a.data_ptr<int32_t>(),
      pair_bins_b.data_ptr<int32_t>(),
      pair_count,
      static_cast<float>(margin_factor),
      static_cast<float>(alpha),
      static_cast<float>(scale),
      grad.data_ptr<float>());
  AT_CUDA_CHECK(cudaGetLastError());
  return grad;
}
