#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace {
constexpr float kEps = 1e-8f;
constexpr float kAreaEps = 1e-12f;
// Kernel configuration for optimal GPU utilization
// TILE_SIZE: Number of cells processed per tile (larger = better memory bandwidth)
// BLOCK_THREADS: Threads per block (more = better parallelism, up to hardware limits)
// Shared memory usage: ~10.1 KB per block (well within 48KB/64KB limits)
constexpr int TILE_SIZE = 512;
constexpr int BLOCK_THREADS = 1024;

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

}  // namespace

// Forward kernel: Computes overlap loss for all cell pairs
// Uses grid-stride loop to process multiple bin-pairs per block
// All threads participate in sync points, even for invalid pairs
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
    int num_bins_x,
    int num_bins_y,
    int total_cells,
    float margin_factor,
    float alpha,
    float scale,
    float* __restrict__ loss_out) {
  
  // Shared memory for block reduction (must be at kernel scope)
  // For BLOCK_THREADS=1024: 1024/32 = 32 warps
  __shared__ float block_reduce_smem[BLOCK_THREADS / 32];
  
  // Shared memory for tile loading
  __shared__ float sh_pos_x[TILE_SIZE];
  __shared__ float sh_pos_y[TILE_SIZE];
  __shared__ float sh_w[TILE_SIZE];
  __shared__ float sh_h[TILE_SIZE];
  __shared__ float sh_area[TILE_SIZE];

  // Grid-stride loop: each thread block handles multiple pairs
  // This allows launching a limited number of blocks (e.g. 65535) even for millions of pairs
  for (int pair_idx = blockIdx.x; pair_idx < pair_count; pair_idx += gridDim.x) {
    // Reset thread local accumulator for this pair
  float thread_loss = 0.0f;
    bool pair_valid = true;

    // Bounds checking: Use flag instead of continue to ensure all threads reach sync points
    // Ensures all threads participate in __syncthreads() calls
    int bin_a = pair_bins_a[pair_idx];
    int bin_b = pair_bins_b[pair_idx];
    
    if (bin_a < 0 || bin_b < 0) pair_valid = false;
    
    int total_bins = num_bins_x * num_bins_y;
    if (pair_valid && (bin_a >= total_bins || bin_b >= total_bins)) pair_valid = false;
    
    int start_a = 0, count_a = 0, start_b = 0, count_b = 0;
    if (pair_valid) {
      start_a = bin_starts[bin_a];
      count_a = bin_counts[bin_a];
      start_b = bin_starts[bin_b];
      count_b = bin_counts[bin_b];
      if (count_a == 0 || count_b == 0 || count_a < 0 || count_b < 0) pair_valid = false;
      
      // Safety check for corruption
      if (pair_valid && (count_a > 50000 || count_b > 50000)) pair_valid = false;
      
      // Bounds check for start indices
      if (pair_valid && (start_a < 0 || start_a >= total_cells || start_b < 0 || start_b >= total_cells)) pair_valid = false;
      if (pair_valid && (start_a + count_a > total_cells || start_b + count_b > total_cells)) pair_valid = false;
    }

    bool same_bin = (pair_valid && bin_a == bin_b);

    // Process pairs only if valid
    // When pair_valid is false, threads skip processing but still participate
    // in reduction (with thread_loss = 0.0f) to ensure all threads reach sync points
    if (pair_valid) {
      int max_iterations = (count_a + blockDim.x - 1) / blockDim.x;
  for (int iter = 0; iter < max_iterations; ++iter) {
    int i_local = iter * blockDim.x + threadIdx.x;
    bool valid_i = (i_local < count_a);
    
    float xi = 0.0f, yi = 0.0f, wi = 0.0f, hi = 0.0f, areai = 0.0f;
        bool has_valid_cell = false;
    if (valid_i) {
          int idx = start_a + i_local;
          if (idx >= 0 && idx < total_cells) {
            int cell_i = sorted_indices[idx];
            if (cell_i >= 0 && cell_i < total_cells) {
      xi = positions[cell_i * 2];
      yi = positions[cell_i * 2 + 1];
      wi = widths[cell_i];
      hi = heights[cell_i];
      areai = areas[cell_i];
              has_valid_cell = true;
            }
          }
    }

        int max_j_tiles = (count_b + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile_idx = 0; tile_idx < max_j_tiles; ++tile_idx) {
      int j_tile = tile_idx * TILE_SIZE;
      
          // Load tile into shared memory
          // Only threads < TILE_SIZE load data, but ALL threads must reach sync point
      if (threadIdx.x < TILE_SIZE) {
        int load_idx = j_tile + threadIdx.x;
        if (load_idx < count_b) {
              int idx = start_b + load_idx;
              if (idx >= 0 && idx < total_cells) {
                int cell_j = sorted_indices[idx];
                if (cell_j >= 0 && cell_j < total_cells) {
          sh_pos_x[threadIdx.x] = positions[cell_j * 2];
          sh_pos_y[threadIdx.x] = positions[cell_j * 2 + 1];
          sh_w[threadIdx.x] = widths[cell_j];
          sh_h[threadIdx.x] = heights[cell_j];
          sh_area[threadIdx.x] = areas[cell_j];
                } else {
                  sh_pos_x[threadIdx.x] = 0.0f; sh_pos_y[threadIdx.x] = 0.0f; sh_w[threadIdx.x] = 0.0f; sh_h[threadIdx.x] = 0.0f; sh_area[threadIdx.x] = 0.0f;
                }
              } else {
                sh_pos_x[threadIdx.x] = 0.0f; sh_pos_y[threadIdx.x] = 0.0f; sh_w[threadIdx.x] = 0.0f; sh_h[threadIdx.x] = 0.0f; sh_area[threadIdx.x] = 0.0f;
              }
            } else {
              // Out of bounds for this tile
              sh_pos_x[threadIdx.x] = 0.0f; sh_pos_y[threadIdx.x] = 0.0f; sh_w[threadIdx.x] = 0.0f; sh_h[threadIdx.x] = 0.0f; sh_area[threadIdx.x] = 0.0f;
            }
          } else {
            // Threads >= TILE_SIZE don't load, but they still participate in sync
            // No action needed - threads wait at __syncthreads()
          }
          
          // All threads (including those >= TILE_SIZE) must reach this sync point
      __syncthreads();

          // Process tile
          if (valid_i && has_valid_cell) {
        int tile_count = min(TILE_SIZE, count_b - j_tile);
        for (int j_local = 0; j_local < tile_count; ++j_local) {
          int global_j = j_tile + j_local;
              if (same_bin && global_j <= i_local) continue;
              
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

              // Only apply margin when there's actual overlap
              // Tightened threshold: only apply margin when cells are actually overlapping
              float overlap_x_base = min_sep_x - abs_dx;
              float overlap_y_base = min_sep_y - abs_dy;
              float margin_x = margin * (overlap_x_base > 0.0f ? 1.0f : 0.0f);
              float margin_y = margin * (overlap_y_base > 0.0f ? 1.0f : 0.0f);
              float overlap_x_raw = overlap_x_base + margin_x;
              float overlap_y_raw = overlap_y_base + margin_y;

          float overlap_x = softplus_scaled(overlap_x_raw, alpha);
          float overlap_y = softplus_scaled(overlap_y_raw, alpha);
          float raw_area = overlap_x * overlap_y;

              if (raw_area > kEps) {
          float overlap_area = fmaxf(raw_area, kEps);
          float repulsion = sqrtf(fmaxf(areai * areaj, kAreaEps));
          float penalty = repulsion * powf(overlap_area, 2.5f);
          if (isfinite(penalty)) {
            thread_loss += penalty;
          }
        }
      }
          }
          
          // All threads must reach this sync after processing each tile
      __syncthreads();
    }
  }
    }
    // If pair_valid is false, thread_loss remains 0.0f and we skip to reduction

    // All threads must participate in reduction for each pair
    // This happens for both valid and invalid pairs (invalid pairs have thread_loss = 0.0f)
    // Step 1: Warp-level reduction (all threads participate)
  float sum = warp_reduce_sum(thread_loss);
  
  // Step 2: Store warp sums (only first thread of each warp writes)
  if ((threadIdx.x & 31) == 0) {
    block_reduce_smem[threadIdx.x >> 5] = sum;
  }
    // All threads must reach this sync point
  __syncthreads();
  
    // Step 3: Final reduction
    // warp_reduce_sum uses __shfl_down_sync which requires all 32 threads in a warp
    // to participate. In the final reduction, only BLOCK_THREADS/32 threads have data.
    // Thread 0 manually sums the warp sums from shared memory to avoid requiring
    // all threads to participate in the warp shuffle.
    if (threadIdx.x == 0) {
      float final_sum = 0.0f;
      for (int i = 0; i < BLOCK_THREADS / 32; ++i) {
        final_sum += block_reduce_smem[i];
      }
      atomicAdd(loss_out, final_sum);
    }
    
    // All threads must reach this final sync point
    // Threads that didn't participate in step 3 still need to sync
  __syncthreads();
    
    // Reset thread_loss for next pair in grid-stride loop
    thread_loss = 0.0f;
  }
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
    int num_bins_x,
    int num_bins_y,
    int total_cells,
    float margin_factor,
    float alpha,
    float scale,
    float* __restrict__ grad_positions) {
  
  // Shared memory for tile loading
  __shared__ float sh_pos_x[TILE_SIZE];
  __shared__ float sh_pos_y[TILE_SIZE];
  __shared__ float sh_w[TILE_SIZE];
  __shared__ float sh_h[TILE_SIZE];
  __shared__ float sh_area[TILE_SIZE];

  // Grid-stride loop: limit blocks to max ~65k
  for (int pair_idx = blockIdx.x; pair_idx < pair_count; pair_idx += gridDim.x) {
    // Use flag instead of continue to ensure all threads participate in sync points
    bool pair_valid = true;
    
    // Bounds checking
    int bin_a = pair_bins_a[pair_idx];
    int bin_b = pair_bins_b[pair_idx];
    
    if (bin_a < 0 || bin_b < 0) pair_valid = false;
    
    int total_bins = num_bins_x * num_bins_y;
    if (pair_valid && (bin_a >= total_bins || bin_b >= total_bins)) pair_valid = false;
    
    int start_a = 0, count_a = 0, start_b = 0, count_b = 0;
    if (pair_valid) {
      start_a = bin_starts[bin_a];
      count_a = bin_counts[bin_a];
      start_b = bin_starts[bin_b];
      count_b = bin_counts[bin_b];
      if (count_a == 0 || count_b == 0 || count_a < 0 || count_b < 0) pair_valid = false;
      
      // Safety check
      if (pair_valid && (count_a > 50000 || count_b > 50000)) pair_valid = false;
      
      // Bounds check for start indices
      if (pair_valid && (start_a < 0 || start_a >= total_cells || start_b < 0 || start_b >= total_cells)) pair_valid = false;
      if (pair_valid && (start_a + count_a > total_cells || start_b + count_b > total_cells)) pair_valid = false;
    }

    bool same_bin = (pair_valid && bin_a == bin_b);

    // Ensure all threads participate even for invalid pairs
    int max_iterations = pair_valid ? ((count_a + blockDim.x - 1) / blockDim.x) : 0;
  for (int iter = 0; iter < max_iterations; ++iter) {
    int i_local = iter * blockDim.x + threadIdx.x;
    bool valid_i = (i_local < count_a);
    
    int cell_i = -1;
    float xi = 0.0f, yi = 0.0f, wi = 0.0f, hi = 0.0f, areai = 0.0f;
      bool has_valid_cell = false;
      
    if (valid_i) {
        int idx = start_a + i_local;
        if (idx >= 0 && idx < total_cells) {
          cell_i = sorted_indices[idx];
          if (cell_i >= 0 && cell_i < total_cells) {
      xi = positions[cell_i * 2];
      yi = positions[cell_i * 2 + 1];
      wi = widths[cell_i];
      hi = heights[cell_i];
      areai = areas[cell_i];
            has_valid_cell = true;
          }
        }
    }

      int max_j_tiles = (count_b + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile_idx = 0; tile_idx < max_j_tiles; ++tile_idx) {
      int j_tile = tile_idx * TILE_SIZE;
      
        // Load tile into shared memory
      if (threadIdx.x < TILE_SIZE) {
        int load_idx = j_tile + threadIdx.x;
        if (load_idx < count_b) {
            int idx = start_b + load_idx;
            if (idx >= 0 && idx < total_cells) {
              int cell_j = sorted_indices[idx];
              if (cell_j >= 0 && cell_j < total_cells) {
          sh_pos_x[threadIdx.x] = positions[cell_j * 2];
          sh_pos_y[threadIdx.x] = positions[cell_j * 2 + 1];
          sh_w[threadIdx.x] = widths[cell_j];
          sh_h[threadIdx.x] = heights[cell_j];
          sh_area[threadIdx.x] = areas[cell_j];
              } else {
                sh_pos_x[threadIdx.x] = 0.0f; sh_pos_y[threadIdx.x] = 0.0f; sh_w[threadIdx.x] = 0.0f; sh_h[threadIdx.x] = 0.0f; sh_area[threadIdx.x] = 0.0f;
              }
            } else {
              sh_pos_x[threadIdx.x] = 0.0f; sh_pos_y[threadIdx.x] = 0.0f; sh_w[threadIdx.x] = 0.0f; sh_h[threadIdx.x] = 0.0f; sh_area[threadIdx.x] = 0.0f;
            }
          }
        }
      __syncthreads();

        // Process tile
        if (valid_i && has_valid_cell) {
        int tile_count = min(TILE_SIZE, count_b - j_tile);
        for (int j_local = 0; j_local < tile_count; ++j_local) {
          int global_j = j_tile + j_local;
            if (same_bin && global_j <= i_local) continue;
            
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

            // Only apply margin when there's actual overlap
            // Tightened threshold: only apply margin when cells are actually overlapping
            float overlap_x_base = min_sep_x - abs_dx;
            float overlap_y_base = min_sep_y - abs_dy;
            float margin_x = margin * (overlap_x_base > 0.0f ? 1.0f : 0.0f);
            float margin_y = margin * (overlap_y_base > 0.0f ? 1.0f : 0.0f);
            float overlap_x_raw = overlap_x_base + margin_x;
            float overlap_y_raw = overlap_y_base + margin_y;

          float overlap_x = softplus_scaled(overlap_x_raw, alpha);
          float overlap_y = softplus_scaled(overlap_y_raw, alpha);
          float raw_area = overlap_x * overlap_y;

            if (raw_area > kEps) {
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

              // Access cell_j from global memory since we need its index
              int idx_j = start_b + j_tile + j_local;
              if (idx_j >= 0 && idx_j < total_cells) { 
                 int cell_j_global = sorted_indices[idx_j];
                 
                 if (cell_j_global >= 0 && cell_j_global < total_cells) {
          if (grad_x_i != 0.0f) {
            atomicAdd(&grad_positions[cell_i * 2], grad_x_i);
                      atomicAdd(&grad_positions[cell_j_global * 2], grad_x_j);
          }
          if (grad_y_i != 0.0f) {
            atomicAdd(&grad_positions[cell_i * 2 + 1], grad_y_i);
                      atomicAdd(&grad_positions[cell_j_global * 2 + 1], grad_y_j);
          }
        }
      }
            }
          }
        }
      __syncthreads();
      }
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

  // Grid-stride loop setup: Limit blocks to prevent GPU scheduler overload
  // Each block processes multiple pairs via grid-stride loop, enabling parallelism
  // while keeping the number of launched blocks manageable
  const int MAX_BLOCKS = 65535;
  int num_blocks;
  if (pair_count > 100000) {
    // Very large problems: use maximum blocks for maximum parallelism
    num_blocks = MAX_BLOCKS;
  } else if (pair_count > 10000) {
    // Large problems: allocate blocks based on pair count
    num_blocks = std::min((int)(pair_count / 5), MAX_BLOCKS);
    num_blocks = std::max(num_blocks, 10000);
  } else if (pair_count > 1000) {
    // Medium problems: allocate blocks based on pair count
    num_blocks = std::min((int)(pair_count / 2), MAX_BLOCKS);
    num_blocks = std::max(num_blocks, 5000);
  } else {
    // Small problems: use all pairs as blocks (or close to it)
    num_blocks = std::min((int)pair_count, MAX_BLOCKS);
  }
  const dim3 blocks(num_blocks);
  const dim3 threads(BLOCK_THREADS);

  // Safety check: Validate pair count to prevent memory issues
  if (pair_count <= 0 || pair_count > 5000000) { 
    return loss.squeeze();
  }
  
  int total_cells = positions.size(0);
  
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
      static_cast<int>(num_bins_x),
      static_cast<int>(num_bins_y),
      total_cells,
      static_cast<float>(margin_factor),
      static_cast<float>(alpha),
      static_cast<float>(scale),
      loss.data_ptr<float>());
  
  // Check for launch errors immediately (non-blocking)
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "CUDA kernel launch error: ", cudaGetErrorString(err));
  }
  
  // Synchronize to ensure kernel completes before returning
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "CUDA kernel execution error: ", cudaGetErrorString(err));
  }
  
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

  // Grid-stride loop setup: same strategy as forward kernel
  const int MAX_BLOCKS = 65535;
  int num_blocks;
  if (pair_count > 100000) {
    // Very large problems: use maximum blocks for maximum parallelism
    num_blocks = MAX_BLOCKS;
  } else if (pair_count > 10000) {
    // Large problems: allocate blocks based on pair count
    num_blocks = std::min((int)(pair_count / 5), MAX_BLOCKS);
    num_blocks = std::max(num_blocks, 10000);
  } else if (pair_count > 1000) {
    // Medium problems: allocate blocks based on pair count
    num_blocks = std::min((int)(pair_count / 2), MAX_BLOCKS);
    num_blocks = std::max(num_blocks, 5000);
  } else {
    // Small problems: use all pairs as blocks (or close to it)
    num_blocks = std::min((int)pair_count, MAX_BLOCKS);
  }

  const dim3 blocks(num_blocks);
  const dim3 threads(BLOCK_THREADS);

  if (pair_count <= 0 || pair_count > 5000000) {
    return grad;
  }
  
  int total_cells = positions.size(0);
  
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
      static_cast<int>(num_bins_x),
      static_cast<int>(num_bins_y),
      total_cells,
      static_cast<float>(margin_factor),
      static_cast<float>(alpha),
      static_cast<float>(scale),
      grad.data_ptr<float>());
  
  // Check for launch errors immediately (non-blocking)
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "CUDA kernel launch error: ", cudaGetErrorString(err));
  }
  
  // Explicitly synchronize to ensure kernel completes and catch any errors
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "CUDA kernel execution error: ", cudaGetErrorString(err));
  }
  
  return grad;
}
