# Overlap Loss Algorithm: Complete Technical Breakdown

## Table of Contents
1. [Mathematical Formulation & Intuition](#1-mathematical-formulation--intuition)
2. [Python-Level Algorithm](#2-python-level-algorithm)
3. [Optimization Path: Chunking & Spatial Hashing](#3-optimization-path-chunking--spatial-hashing)
4. [Translation Into CUDA](#4-translation-into-cuda)
5. [Why Multiple CUDA Files Were Needed](#5-why-multiple-cuda-files-were-needed)
6. [End-to-End Implementation Walkthrough](#6-end-to-end-implementation-walkthrough)
7. [Future CUDA-Writing Guidance](#7-future-cuda-writing-guidance)
8. [Code Examples](#8-code-examples)

---

## 1. Mathematical Formulation & Intuition

### 1.1 Core Formula

The overlap loss penalizes overlapping cells in VLSI placement. For two cells `i` and `j` with positions `(xi, yi)`, `(xj, yj)` and dimensions `(wi, hi)`, `(wj, hj)`, the loss is:

```
L_overlap = Σ_{i<j} w_ij · (overlap_area_ij)^2.5
```

Where:
- `w_ij = sqrt(area_i · area_j)` is the repulsion strength (larger cells have stronger repulsion)
- `overlap_area_ij = max(0, overlap_x) · max(0, overlap_y)` is the 2D overlap area
- The exponent `2.5` provides a super-quadratic penalty that grows rapidly with overlap size

### 1.2 Overlap Computation

The overlap in each dimension is computed as:

```
overlap_x = max(0, min_sep_x - |xi - xj| + margin_x)
overlap_y = max(0, min_sep_y - |yi - yj| + margin_y)
```

Where:
- `min_sep_x = 0.5 · (wi + wj)` is the minimum separation needed to avoid overlap
- `min_sep_y = 0.5 · (hi + hj)` is the minimum separation needed to avoid overlap
- `margin = margin_factor · min(min(wi, hi), min(wj, hj))` is an adaptive margin
- `margin_x = margin · (overlap_x_base > 0)` (only applied when there's actual overlap)

### 1.3 StrongFastSigmoid Surrogate Gradient

Instead of using a hard `max(0, x)`, we use a smooth surrogate gradient function `StrongFastSigmoid`:

**Forward pass:**
```
StrongFastSigmoid(x, α, scale) = softplus(α · x) / α
```

Where `softplus(x) = log(1 + exp(x))` is a smooth approximation of ReLU.

**Backward pass (gradient):**
```
grad = scale · sigmoid(α · x) · (1 + 2·tanh(x/2))
```

This provides:
- **Magnitude-aware gradients**: Larger overlaps get stronger gradients
- **Smooth activation**: Avoids discontinuities that can cause training instability
- **Fast computation**: Uses piecewise linear approximation for 3-5x speedup

### 1.4 Intuition

**Why this works:**
1. **Repulsion strength `w_ij`**: Larger cells (macros) have more area, so they should push harder against overlaps. The square root provides a balance between area and gradient magnitude.

2. **Power law penalty `(overlap_area)^2.5`**: The super-quadratic exponent means:
   - Small overlaps (0.1 area) → penalty ≈ 0.003
   - Medium overlaps (1.0 area) → penalty ≈ 1.0
   - Large overlaps (10.0 area) → penalty ≈ 316
   
   This creates a "soft constraint" that becomes increasingly hard as overlap grows.

3. **Adaptive margin**: The margin starts high (5-7% of cell size) to force separation, then gradually reduces to near-zero (1e-8) for fine-tuning. This prevents cells from getting "stuck" in near-overlap states.

4. **Smooth surrogate gradient**: Unlike hard ReLU, the smooth activation provides gradients even when cells are very close but not overlapping, allowing fine-grained optimization.

### 1.5 Gradient Flow

The gradient with respect to cell positions is:

```
∂L/∂xi = -Σ_j w_ij · 2.5 · (overlap_area)^1.5 · overlap_y · grad_factor_x · sign(xi - xj)
∂L/∂yi = -Σ_j w_ij · 2.5 · (overlap_area)^1.5 · overlap_x · grad_factor_y · sign(yi - yj)
```

Where:
- `grad_factor_x = StrongFastSigmoid_grad(overlap_x_raw, α, scale)`
- `sign(xi - xj)` determines the direction (cell i should move away from cell j)

The gradient pushes overlapping cells apart, with strength proportional to the overlap area and repulsion weight.

---

## 2. Python-Level Algorithm

### 2.1 Naive Implementation (O(N²))

The simplest implementation checks all pairs:

```python
def overlap_loss_naive(cell_features):
    N = cell_features.shape[0]
    x, y, w, h, area = extract_features(cell_features)
    
    total_loss = 0.0
    for i in range(N):
        for j in range(i + 1, N):  # Upper triangle only
            # Compute overlap
            dx = abs(x[i] - x[j])
            dy = abs(y[i] - y[j])
            min_sep_x = 0.5 * (w[i] + w[j])
            min_sep_y = 0.5 * (h[i] + h[j])
            
            overlap_x = max(0, min_sep_x - dx)
            overlap_y = max(0, min_sep_y - dy)
            overlap_area = overlap_x * overlap_y
            
            # Penalty
            repulsion = sqrt(area[i] * area[j])
            penalty = repulsion * (overlap_area ** 2.5)
            total_loss += penalty
    
    return total_loss
```

**Complexity**: O(N²) pairs × O(1) computation = **O(N²)** total

**Bottlenecks**:
1. **Pairwise comparisons**: For N=100,000 cells, we need 5 billion pairs
2. **Sequential loops**: Python loops are slow, no vectorization
3. **Memory access**: Random access patterns, poor cache locality
4. **Branching**: `if` statements in inner loop cause pipeline stalls

### 2.2 Vectorized PyTorch Implementation

PyTorch allows vectorized operations on GPU:

```python
def overlap_loss_vectorized(cell_features, margin_factor, alpha, scale):
    N = cell_features.shape[0]
    x, y, w, h, area = extract_features(cell_features)
    
    # Broadcast to [N, N] matrices
    xi, yi = x.unsqueeze(1), y.unsqueeze(1)  # [N, 1]
    xj, yj = x.unsqueeze(0), y.unsqueeze(0)  # [1, N]
    wi, hi = w.unsqueeze(1), h.unsqueeze(1)  # [N, 1]
    wj, hj = w.unsqueeze(0), h.unsqueeze(0)  # [1, N]
    
    # Vectorized pairwise computation
    dx_abs = torch.abs(xi - xj)  # [N, N]
    dy_abs = torch.abs(yi - yj)  # [N, N]
    
    min_sep_x = 0.5 * (wi + wj)  # [N, N]
    min_sep_y = 0.5 * (hi + hj)  # [N, N]
    
    overlap_x_base = min_sep_x - dx_abs
    overlap_y_base = min_sep_y - dy_abs
    
    # Apply StrongFastSigmoid
    overlap_x = strong_fast_sigmoid(overlap_x_base, scale, alpha)
    overlap_y = strong_fast_sigmoid(overlap_y_base, scale, alpha)
    
    # Upper triangle mask (i < j)
    valid_mask = torch.triu(torch.ones(N, N, device=device), diagonal=1)
    
    # Compute penalty
    overlap_area = torch.clamp(overlap_x, min=0) * torch.clamp(overlap_y, min=0)
    repulsion = torch.sqrt(area.unsqueeze(1) * area.unsqueeze(0))
    penalty = repulsion * (overlap_area ** 2.5) * valid_mask
    
    return penalty.sum()
```

**Improvements**:
- **Vectorization**: All operations are element-wise on [N, N] tensors
- **GPU acceleration**: PyTorch operations run on GPU automatically
- **No Python loops**: Everything is tensor operations

**Remaining bottlenecks**:
1. **Memory**: O(N²) tensors (for N=100k, that's 40GB for float32)
2. **Computation**: Still O(N²) operations, just faster on GPU
3. **Memory bandwidth**: Loading/storing [N, N] tensors is expensive

### 2.3 Chunked Processing

To handle large N, we process in chunks:

```python
def overlap_loss_chunked(cell_features, chunk_size=5000):
    N = cell_features.shape[0]
    total_loss = 0.0
    
    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        i_chunk = cell_features[i_start:i_end]
        
        for j_start in range(i_start + 1, N, chunk_size):
            j_end = min(j_start + chunk_size, N)
            j_chunk = cell_features[j_start:j_end]
            
            # Process chunk pair
            chunk_loss = compute_overlap_chunk(i_chunk, j_chunk)
            total_loss += chunk_loss
    
    return total_loss
```

**Benefits**:
- **Memory**: Only [chunk_size, chunk_size] tensors in memory
- **GPU utilization**: Smaller tensors fit in GPU memory
- **Flexibility**: Can adjust chunk size based on GPU memory

**Trade-offs**:
- **Python overhead**: Loop iterations add overhead
- **Still O(N²)**: We still check all pairs, just in smaller chunks
- **Cache misses**: Loading different chunks repeatedly

---

## 3. Optimization Path: Chunking & Spatial Hashing

### 3.1 Why Pairwise Operations Explode

For N cells, the number of pairs is:
```
pairs = N · (N - 1) / 2 ≈ N²/2
```

**Examples**:
- N = 1,000 → 500,000 pairs
- N = 10,000 → 50 million pairs
- N = 100,000 → 5 billion pairs
- N = 1,000,000 → 500 billion pairs

Even with GPU acceleration, 5 billion pairs is computationally expensive.

### 3.2 Spatial Hashing: O(N²) → O(N)

**Key insight**: Most cell pairs are far apart and don't overlap. We only need to check nearby pairs.

**Spatial hashing** bins cells by their spatial location:

```
1. Divide space into a grid of bins (e.g., 50×50 bins)
2. Assign each cell to a bin based on its position
3. Only check pairs within the same bin or adjacent bins
```

**Complexity reduction**:
- **Without hashing**: Check all N² pairs
- **With hashing**: Check only pairs in nearby bins
- **Average case**: If cells are uniformly distributed, each bin has ~N/bins cells
- **Pairs per bin**: ~(N/bins)²
- **Total bins**: ~bins
- **Total pairs**: ~bins · (N/bins)² = N²/bins

For bins ≈ N (one bin per cell on average), we get **O(N)** complexity!

### 3.3 Spatial Hashing Algorithm

```python
def build_spatial_index(positions, widths, heights, bin_size):
    """Build spatial hash index."""
    x, y = positions[:, 0], positions[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Create grid of bins
    num_bins_x = int((x_max - x_min) / bin_size) + 1
    num_bins_y = int((y_max - y_min) / bin_size) + 1
    
    # Assign cells to bins
    bin_x = ((x - x_min) / bin_size).long().clamp(0, num_bins_x - 1)
    bin_y = ((y - y_min) / bin_size).long().clamp(0, num_bins_y - 1)
    bin_idx = bin_y * num_bins_x + bin_x  # Flattened index
    
    # Sort cells by bin for efficient access
    sorted_indices = torch.argsort(bin_idx)
    sorted_bin_idx = bin_idx[sorted_indices]
    
    # Find bin boundaries
    unique_bins, bin_counts = torch.unique_consecutive(sorted_bin_idx, return_counts=True)
    bin_starts = torch.cat([torch.tensor([0]), bin_counts.cumsum(0)[:-1]])
    
    return SpatialIndex(
        cell_bins=bin_idx,
        bin_starts=bin_starts,
        bin_counts=bin_counts,
        sorted_indices=sorted_indices,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y
    )
```

### 3.4 Processing Bin Pairs

```python
def process_bin_pairs(spatial_index, cell_features):
    """Process only pairs within same or adjacent bins."""
    total_loss = 0.0
    
    for bin_a in unique_bins:
        cells_a = get_cells_in_bin(bin_a, spatial_index)
        
        # Check same bin and 8 adjacent bins
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                bin_b = get_neighbor_bin(bin_a, dx, dy)
                cells_b = get_cells_in_bin(bin_b, spatial_index)
                
                # Process pairs between cells_a and cells_b
                chunk_loss = compute_overlap_chunk(cells_a, cells_b)
                total_loss += chunk_loss
    
    return total_loss
```

**Filtering**:
- **Same bin**: All pairs within the bin
- **Adjacent bins**: Only pairs that could overlap (cells near bin boundaries)
- **Distant bins**: Skipped entirely (no possible overlap)

### 3.5 Performance Intuition

**Cache behavior**:
- Cells in the same bin are spatially close → better cache locality
- Sequential access of sorted indices → better memory bandwidth

**Branch reduction**:
- Early exit for distant bins → fewer branches
- Vectorized operations within bins → fewer conditionals

**Vectorization friendliness**:
- Processing all pairs in a bin at once → large tensor operations
- Better GPU utilization → fewer kernel launches

**Memory pressure**:
- Only active bins in memory → lower peak memory
- Chunked processing within bins → fits in GPU memory

### 3.6 Optimal Bin Size

The bin size is a trade-off:
- **Too small**: Many bins, more overhead, but very accurate
- **Too large**: Few bins, less overhead, but may miss some overlaps

**Heuristic**:
```python
bin_size = max(mean_cell_dimension * 4, max_cell_dimension, 1.0)
```

This ensures:
- Bins are large enough to contain cells and their neighbors
- Not too large to lose the filtering benefit
- Adapts to cell size distribution

---

## 4. Translation Into CUDA

### 4.1 Kernel Design

The CUDA kernel processes bin pairs using a **grid-stride loop** pattern:

```cuda
__global__ void overlap_forward_kernel(
    const float* positions,      // [N, 2] flattened
    const float* widths,          // [N]
    const float* heights,         // [N]
    const float* areas,           // [N]
    const int32_t* bin_starts,    // [num_bins]
    const int32_t* bin_counts,    // [num_bins]
    const int32_t* sorted_indices, // [N]
    const int32_t* pair_bins_a,   // [num_pairs]
    const int32_t* pair_bins_b,  // [num_pairs]
    int pair_count,
    float margin_factor,
    float alpha,
    float scale,
    float* loss_out) {
    
    // Shared memory for tile loading
    __shared__ float sh_pos_x[TILE_SIZE];
    __shared__ float sh_pos_y[TILE_SIZE];
    __shared__ float sh_w[TILE_SIZE];
    __shared__ float sh_h[TILE_SIZE];
    __shared__ float sh_area[TILE_SIZE];
    
    // Grid-stride loop: each block processes multiple bin pairs
    for (int pair_idx = blockIdx.x; pair_idx < pair_count; pair_idx += gridDim.x) {
        float thread_loss = 0.0f;
        
        // Get bin pair
        int bin_a = pair_bins_a[pair_idx];
        int bin_b = pair_bins_b[pair_idx];
        
        // Get cells in bins
        int start_a = bin_starts[bin_a];
        int count_a = bin_counts[bin_a];
        int start_b = bin_starts[bin_b];
        int count_b = bin_counts[bin_b];
        
        // Process cells in bin_a against cells in bin_b
        for (int i = threadIdx.x; i < count_a; i += blockDim.x) {
            // Load cell i
            int cell_i = sorted_indices[start_a + i];
            float xi = positions[cell_i * 2];
            float yi = positions[cell_i * 2 + 1];
            // ... load other features
            
            // Process against all cells in bin_b (tiled)
            for (int tile = 0; tile < (count_b + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
                // Load tile into shared memory
                if (threadIdx.x < TILE_SIZE) {
                    int j = tile * TILE_SIZE + threadIdx.x;
                    if (j < count_b) {
                        int cell_j = sorted_indices[start_b + j];
                        sh_pos_x[threadIdx.x] = positions[cell_j * 2];
                        // ... load other features
                    }
                }
                __syncthreads();
                
                // Process tile
                for (int j_local = 0; j_local < TILE_SIZE && (tile * TILE_SIZE + j_local) < count_b; ++j_local) {
                    float xj = sh_pos_x[j_local];
                    // ... compute overlap
                    thread_loss += penalty;
                }
                __syncthreads();
            }
        }
        
        // Reduce thread_loss to block_loss, then atomic add to global loss
        // ... reduction code ...
    }
}
```

### 4.2 Memory Layout

**Global memory** (device memory):
- `positions`, `widths`, `heights`, `areas`: Cell features, read-only
- `bin_starts`, `bin_counts`, `sorted_indices`: Spatial index, read-only
- `pair_bins_a`, `pair_bins_b`: Bin pairs to process, read-only
- `loss_out`: Output loss, written atomically

**Shared memory** (per-block, fast):
- `sh_pos_x`, `sh_pos_y`, `sh_w`, `sh_h`, `sh_area`: Tile of cell j features
- `block_reduce_smem`: Temporary storage for reduction

**Constant memory** (cached, read-only):
- Kernel parameters: `margin_factor`, `alpha`, `scale`, `TILE_SIZE`, etc.

### 4.3 Coalesced Memory Access

**Coalescing** means threads in a warp (32 threads) access consecutive memory locations:

```cuda
// GOOD: Coalesced access (consecutive indices)
int cell_i = sorted_indices[start_a + i];  // i = 0, 1, 2, ..., 31
float xi = positions[cell_i * 2];        // Accesses consecutive memory

// BAD: Random access (non-coalesced)
int cell_i = random_indices[i];           // Random indices
float xi = positions[cell_i * 2];        // Scattered memory access
```

**Our implementation**:
- `sorted_indices` ensures consecutive access within bins
- Loading tiles into shared memory allows coalesced reads
- Processing tiles sequentially improves cache reuse

### 4.4 Avoiding Warp Divergence

**Warp divergence** occurs when threads in a warp take different execution paths:

```cuda
// BAD: Divergence (some threads skip, others continue)
if (i < count_a) {
    // Process cell i
}

// GOOD: All threads participate, use flags
bool valid_i = (i < count_a);
if (valid_i) {
    // Process cell i
}
// All threads continue to sync point
__syncthreads();
```

**Our implementation**:
- All threads participate in `__syncthreads()` calls
- Use flags (`pair_valid`, `valid_i`) instead of `continue`/`break`
- Ensure all threads reach reduction code

### 4.5 Parallelization Strategy

**Grid-stride loop**:
- **Problem**: We may have millions of bin pairs, but GPU can only launch ~65k blocks
- **Solution**: Each block processes multiple pairs via grid-stride loop
- **Benefit**: Maximizes GPU utilization while staying within hardware limits

```cuda
// Each block processes multiple pairs
for (int pair_idx = blockIdx.x; pair_idx < pair_count; pair_idx += gridDim.x) {
    // Process this pair
}
```

**Block/thread sizing**:
- **Blocks**: Up to 65,535 blocks (hardware limit)
- **Threads per block**: 1024 threads (optimal for our workload)
- **Tiles**: 512 cells per tile (fits in shared memory, ~10KB)

### 4.6 Error Fixes

#### 4.6.1 Race Conditions / Write Conflicts

**Problem**: Multiple threads writing to the same memory location without synchronization.

**Fix**: Use atomic operations for reduction:

```cuda
// BAD: Race condition
loss_out[0] += thread_loss;  // Multiple threads write simultaneously

// GOOD: Atomic operation
atomicAdd(loss_out, thread_loss);  // Thread-safe addition
```

#### 4.6.2 Out-of-Bounds Indexing

**Problem**: Accessing array indices beyond allocated bounds.

**Fix**: Bounds checking before all array accesses:

```cuda
// BAD: No bounds check
int cell_i = sorted_indices[idx];
float xi = positions[cell_i * 2];

// GOOD: Bounds checking
if (idx >= 0 && idx < total_cells) {
    int cell_i = sorted_indices[idx];
    if (cell_i >= 0 && cell_i < total_cells) {
        float xi = positions[cell_i * 2];
    }
}
```

#### 4.6.3 Misalignment of Tensor Strides

**Problem**: PyTorch tensors may not be contiguous, causing incorrect memory access.

**Fix**: Ensure tensors are contiguous before passing to CUDA:

```python
# Python side
positions = positions.contiguous()
widths = widths.contiguous()
# ... etc
```

#### 4.6.4 Invalid Shared Memory Usage

**Problem**: Shared memory size exceeded (48KB/64KB limit per block).

**Fix**: Calculate shared memory usage:

```cuda
// Shared memory: 5 arrays × TILE_SIZE × 4 bytes
// TILE_SIZE = 512 → 5 × 512 × 4 = 10,240 bytes ≈ 10KB
// Well within 48KB limit
__shared__ float sh_pos_x[TILE_SIZE];  // 2KB
__shared__ float sh_pos_y[TILE_SIZE];  // 2KB
__shared__ float sh_w[TILE_SIZE];     // 2KB
__shared__ float sh_h[TILE_SIZE];     // 2KB
__shared__ float sh_area[TILE_SIZE];  // 2KB
```

#### 4.6.5 Deadlocks Due to Sync Misuse

**Problem**: Not all threads reaching `__syncthreads()` causes deadlock.

**Fix**: Ensure all threads participate in sync points:

```cuda
// BAD: Some threads skip sync
if (threadIdx.x < TILE_SIZE) {
    // Load data
}
__syncthreads();  // Only TILE_SIZE threads reach here, others skip → deadlock

// GOOD: All threads reach sync
if (threadIdx.x < TILE_SIZE) {
    // Load data
} else {
    // Do nothing, but thread still exists
}
__syncthreads();  // All 1024 threads reach here
```

#### 4.6.6 Precision or Dtype Mismatch

**Problem**: Mixing float32 and float64, or incorrect type casting.

**Fix**: Consistent float32 usage:

```cuda
// All computations in float32
float xi = positions[cell_i * 2];  // float32
float margin = margin_factor * min_dim_pair;  // float32

// Explicit casting when needed
float alpha_safe = fmaxf(alpha, 1e-8f);  // float32 constant
```

---

## 5. Why Multiple CUDA Files Were Needed

### 5.1 File Structure

```
cuda_backend/
├── overlap_cuda.cpp          # C++ wrapper (PyTorch interface)
├── overlap_cuda_kernel.cu    # CUDA kernels (forward/backward)
├── overlap_cuda.py           # Python wrapper (autograd)
└── setup_cuda.py             # Build script
```

### 5.2 Separation of Concerns

#### `overlap_cuda_kernel.cu` (CUDA Kernels)
- **Purpose**: Pure CUDA kernel code
- **Contains**: `__global__` and `__device__` functions
- **Why separate**: CUDA compiler (`nvcc`) handles this file differently than C++

#### `overlap_cuda.cpp` (C++ Wrapper)
- **Purpose**: PyTorch tensor interface
- **Contains**: C++ functions that call CUDA kernels
- **Why separate**: C++ compiler (`g++`) compiles this, links with CUDA object files

#### `overlap_cuda.py` (Python Wrapper)
- **Purpose**: Autograd integration and spatial hashing
- **Contains**: `torch.autograd.Function` subclass
- **Why separate**: Python code, not compiled

### 5.3 Forward vs Backward Kernels

**Forward kernel** (`overlap_forward_kernel`):
- Computes loss value: `L = Σ penalties`
- Returns: Scalar loss
- Simpler: Only needs to accumulate penalties

**Backward kernel** (`overlap_backward_kernel`):
- Computes gradients: `∂L/∂positions`
- Returns: Gradient tensor `[N, 2]`
- More complex: Needs to compute and accumulate gradients for each cell

**Why separate**:
- Different memory access patterns (forward: read-only, backward: atomic writes)
- Different reduction operations (forward: sum, backward: per-cell accumulation)
- Easier to optimize each independently

### 5.4 Templated vs Non-Templated

**Our implementation**: Non-templated (explicit float32)

**Why**:
- Simpler compilation (no template instantiation)
- Faster compile times
- Sufficient for our use case (always float32)

**Alternative (templated)**:
```cuda
template<typename T>
__global__ void overlap_forward_kernel(...) {
    // Works with float32, float64, etc.
}
```

**Trade-off**: More flexible but slower compilation and more complex code.

### 5.5 Build System Constraints

**PyTorch extensions** require:
1. C++ source files (`.cpp`) compiled with `g++`
2. CUDA source files (`.cu`) compiled with `nvcc`
3. Linking: C++ object files + CUDA object files → shared library (`.so`)

**Our `setup_cuda.py`**:
```python
CUDAExtension(
    name="overlap_cuda_backend",
    sources=[
        "overlap_cuda.cpp",      # Compiled by g++
        "overlap_cuda_kernel.cu", # Compiled by nvcc
    ],
)
```

**Compilation flow**:
```
overlap_cuda.cpp → g++ → overlap_cuda.o
overlap_cuda_kernel.cu → nvcc → overlap_cuda_kernel.o
overlap_cuda.o + overlap_cuda_kernel.o → linker → overlap_cuda_backend.so
```

### 5.6 Compilation Units for Rebuild Time

**Separate files** allow:
- **Incremental compilation**: Only recompile changed files
- **Parallel compilation**: Compile `.cpp` and `.cu` simultaneously
- **Faster iteration**: Change kernel code without recompiling wrapper

**Single file** would require:
- Recompiling everything on any change
- Slower iteration during development

---

## 6. End-to-End Implementation Walkthrough

### 6.1 Call Flow

```
Python (placement.py)
  ↓
overlap_repulsion_loss() [losses.py]
  ↓
compute_overlap_loss() [overlap_cuda.py]  # If N >= 50000
  ↓
_OverlapCUDAFunction.apply() [overlap_cuda.py]
  ↓
overlap_cuda_backend.forward() [overlap_cuda.cpp]
  ↓
overlap_forward_cuda() [overlap_cuda_kernel.cu]
  ↓
overlap_forward_kernel<<<blocks, threads>>>() [CUDA GPU]
```

### 6.2 Data Flow

**Step 1: Python → C++ (PyTorch tensors)**
```python
# Python side (overlap_cuda.py)
loss = _OverlapCUDAFunction.apply(
    positions.contiguous(),    # [N, 2] float32
    widths.contiguous(),        # [N] float32
    heights.contiguous(),       # [N] float32
    areas.contiguous(),         # [N] float32
    # ... spatial index tensors
)
```

**Step 2: C++ → CUDA (raw pointers)**
```cpp
// C++ side (overlap_cuda.cpp)
overlap_forward_cuda(
    positions.data_ptr<float>(),  // Raw float* pointer
    widths.data_ptr<float>(),
    // ... etc
)
```

**Step 3: CUDA kernel launch**
```cuda
// CUDA side (overlap_cuda_kernel.cu)
overlap_forward_kernel<<<blocks, threads>>>(
    positions,  // Device memory pointer
    // ... etc
)
```

**Step 4: CUDA → C++ (result tensor)**
```cpp
// C++ side
auto loss = torch::zeros({1}, positions.options());
// Kernel writes to loss.data_ptr<float>()
return loss.squeeze();
```

**Step 5: C++ → Python (PyTorch tensor)**
```python
# Python side receives torch.Tensor
return loss  # Can be used in autograd graph
```

### 6.3 Spatial Hashing Setup (Python)

```python
# Build spatial index
bin_size = _compute_bin_size(widths, heights, N)
spatial_index = _build_spatial_index(positions, widths, heights, bin_size)

# Build bin pairs (only adjacent bins)
pair_bins_a, pair_bins_b = _build_bin_pairs(
    spatial_index.bin_counts,
    spatial_index.num_bins_x,
    spatial_index.num_bins_y
)

# Pass to CUDA
loss = _OverlapCUDAFunction.apply(
    positions,
    widths, heights, areas,
    spatial_index.cell_bins,
    spatial_index.bin_starts,
    spatial_index.bin_counts,
    spatial_index.sorted_indices,
    spatial_index.num_bins_x,
    spatial_index.num_bins_y,
    pair_bins_a,
    pair_bins_b,
    margin_factor, alpha, scale
)
```

### 6.4 Kernel Execution (CUDA)

```cuda
// Launch configuration
const int MAX_BLOCKS = 65535;
int num_blocks = min(pair_count / 5, MAX_BLOCKS);  // Adaptive
const dim3 blocks(num_blocks);
const dim3 threads(1024);

// Launch kernel
overlap_forward_kernel<<<blocks, threads>>>(
    positions.data_ptr<float>(),
    // ... all parameters
);

// Synchronize
cudaDeviceSynchronize();
```

### 6.5 Gradient Computation (Backward Pass)

**Autograd integration**:
```python
class _OverlapCUDAFunction(Function):
    @staticmethod
    def forward(ctx, ...):
        loss = overlap_cuda_backend.forward(...)
        ctx.save_for_backward(...)  # Save for backward
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        grad_positions = overlap_cuda_backend.backward(...)
        return grad_positions * grad_output, None, None, ...
```

**Backward kernel**:
- Computes `∂L/∂positions` for each cell
- Uses atomic operations to accumulate gradients
- Returns gradient tensor `[N, 2]`

---

## 7. Future CUDA-Writing Guidance

### 7.1 Planning Kernels for Geometric/Interaction Losses

**Key considerations**:
1. **Spatial locality**: Use spatial hashing/BVH/KD-trees to reduce pairs
2. **Memory access patterns**: Design for coalesced access
3. **Reduction operations**: Plan how to accumulate results (sum, max, etc.)
4. **Scalability**: Use grid-stride loops for large problems

**Design process**:
1. Start with naive O(N²) implementation (validate correctness)
2. Add spatial filtering (spatial hashing, BVH, etc.)
3. Optimize memory access (coalescing, shared memory)
4. Optimize computation (fused operations, fast math)

### 7.2 Avoiding Memory Bottlenecks Early

**Memory hierarchy** (fastest to slowest):
1. **Registers**: Per-thread, fastest
2. **Shared memory**: Per-block, ~10-100x faster than global
3. **L1/L2 cache**: Automatic, helps with coalesced access
4. **Global memory**: Device memory, slowest

**Best practices**:
- **Use shared memory for frequently accessed data**: Load tiles into shared memory
- **Coalesce global memory access**: Threads in warp access consecutive locations
- **Minimize memory transactions**: Fuse operations, reuse loaded data
- **Avoid bank conflicts**: Stride-1 access in shared memory

### 7.3 Deciding Between Optimization Strategies

**Spatial hashing** (our choice):
- **Best for**: Uniform cell distribution, 2D/3D problems
- **Complexity**: O(N) average case
- **Implementation**: Simple grid-based binning
- **When to use**: When cells are roughly uniformly distributed

**BVH (Bounding Volume Hierarchy)**:
- **Best for**: Non-uniform distribution, dynamic scenes
- **Complexity**: O(N log N) build, O(N) query
- **Implementation**: Tree structure, more complex
- **When to use**: When cells cluster in regions

**KD-tree**:
- **Best for**: High-dimensional problems, nearest neighbor queries
- **Complexity**: O(N log N) build, O(log N) query
- **Implementation**: Recursive tree structure
- **When to use**: When you need exact nearest neighbors

**Chunking** (always needed):
- **Best for**: Managing GPU memory limits
- **Complexity**: O(N²) but with memory control
- **Implementation**: Simple loops
- **When to use**: Always, as a fallback or complement to spatial filtering

### 7.4 Validating Correctness Before Optimizing

**Validation strategy**:
1. **Reference implementation**: Naive Python/PyTorch version
2. **Unit tests**: Small test cases (N=10, 100, 1000)
3. **Numerical comparison**: Compare CUDA output to reference (within tolerance)
4. **Gradient checking**: Verify backward pass matches reference
5. **Edge cases**: Empty bins, single cell, all cells in one bin

**Debugging tools**:
- **`cuda-memcheck`**: Detects memory errors
- **`nsight compute`**: Profiler for kernel performance
- **`printf` in kernels**: Debug output (use sparingly, impacts performance)
- **Python assertions**: Validate inputs before CUDA call

### 7.5 Debugging GPU Numerical Errors

**Common issues**:
1. **NaN/Inf**: Check for division by zero, log of negative, etc.
2. **Precision**: Float32 vs float64 mismatches
3. **Non-determinism**: Atomic operations, race conditions
4. **Out-of-bounds**: Array indexing errors

**Debugging steps**:
1. **Add bounds checks**: Validate all array accesses
2. **Check for NaN/Inf**: `if (!isfinite(value)) { printf(...); }`
3. **Compare to reference**: Run same input on CPU, compare outputs
4. **Reduce parallelism**: Test with single thread/block first
5. **Use `cuda-memcheck`**: `cuda-memcheck python script.py`

### 7.6 Structuring Large CUDA Projects

**File organization**:
```
project/
├── kernels/
│   ├── forward.cu
│   ├── backward.cu
│   └── utils.cuh          # Device functions
├── cpp/
│   ├── forward.cpp        # C++ wrappers
│   └── backward.cpp
├── python/
│   └── cuda_module.py     # Python interface
└── setup.py               # Build script
```

**Code organization**:
- **Device functions** (`.cuh`): Reusable `__device__` functions
- **Kernels** (`.cu`): `__global__` functions, one file per kernel
- **C++ wrappers** (`.cpp`): PyTorch interface, one file per operation
- **Python interface** (`.py`): Autograd integration, high-level API

### 7.7 Practical Rules of Thumb

**Warp sizing**:
- Warp = 32 threads (hardware unit)
- Design for warp-aligned access (multiples of 32)
- Avoid warp divergence (all threads in warp take same path)

**Avoiding divergence**:
- Use flags instead of `continue`/`break` in loops
- Ensure all threads reach `__syncthreads()`
- Minimize `if` statements in inner loops

**Shared memory patterns**:
- Load tiles: Multiple threads load data into shared memory
- Process tiles: All threads process shared data
- Sync: `__syncthreads()` between load and process

**Memory alignment**:
- Align arrays to 128 bytes (cache line size)
- Use `__align__(16)` or `__align__(32)` for structs
- PyTorch tensors are already aligned

**Profiling steps**:
1. **`nvprof`**: Basic profiling, identify slow kernels
2. **`nsight compute`**: Detailed analysis, memory access patterns
3. **`nsight systems`**: Full application timeline
4. **PyTorch profiler**: Python-level profiling

**Typical workflow**:
```bash
# 1. Basic profiling
nvprof python script.py

# 2. Detailed analysis
nsight-compute python script.py

# 3. Memory analysis
cuda-memcheck python script.py
```

---

## 8. Code Examples

### 8.1 Simplified Python Algorithm

```python
import torch

def overlap_loss_simplified(cell_features, margin_factor=0.01, alpha=10.0, scale=5.0):
    """
    Simplified overlap loss for demonstration.
    
    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        margin_factor: Margin scaling factor
        alpha: Surrogate gradient sharpness
        scale: Surrogate gradient scale
    
    Returns:
        Scalar loss value
    """
    N = cell_features.shape[0]
    if N <= 1:
        return torch.tensor(0.0, device=cell_features.device, requires_grad=True)
    
    # Extract features
    x = cell_features[:, 2]  # [N]
    y = cell_features[:, 3]  # [N]
    w = cell_features[:, 4]  # [N]
    h = cell_features[:, 5]  # [N]
    area = cell_features[:, 0]  # [N]
    
    # Broadcast to [N, N] matrices
    xi, yi = x.unsqueeze(1), y.unsqueeze(1)  # [N, 1]
    xj, yj = x.unsqueeze(0), y.unsqueeze(0)  # [1, N]
    wi, hi = w.unsqueeze(1), h.unsqueeze(1)  # [N, 1]
    wj, hj = w.unsqueeze(0), h.unsqueeze(0)  # [1, N]
    areai, areaj = area.unsqueeze(1), area.unsqueeze(0)  # [N, 1], [1, N]
    
    # Compute pairwise distances
    dx_abs = torch.abs(xi - xj)  # [N, N]
    dy_abs = torch.abs(yi - yj)  # [N, N]
    
    # Minimum separation
    min_sep_x = 0.5 * (wi + wj)  # [N, N]
    min_sep_y = 0.5 * (hi + hj)  # [N, N]
    
    # Margin
    min_dim_i = torch.minimum(wi, hi)  # [N, 1]
    min_dim_j = torch.minimum(wj, hj)  # [1, N]
    min_dim_pair = torch.minimum(min_dim_i, min_dim_j)  # [N, N]
    margin = margin_factor * min_dim_pair  # [N, N]
    
    # Overlap (with margin)
    overlap_x_base = min_sep_x - dx_abs  # [N, N]
    overlap_y_base = min_sep_y - dy_abs  # [N, N]
    margin_x = margin * (overlap_x_base > 0.0).float()
    margin_y = margin * (overlap_y_base > 0.0).float()
    overlap_x_raw = overlap_x_base + margin_x
    overlap_y_raw = overlap_y_base + margin_y
    
    # Surrogate gradient (simplified: use softplus)
    overlap_x = torch.nn.functional.softplus(overlap_x_raw * alpha) / alpha
    overlap_y = torch.nn.functional.softplus(overlap_y_raw * alpha) / alpha
    
    # Upper triangle mask (i < j)
    valid_mask = torch.triu(torch.ones(N, N, device=cell_features.device, dtype=torch.bool), diagonal=1)
    
    # Compute penalty
    overlap_area = torch.clamp(overlap_x, min=0.0) * torch.clamp(overlap_y, min=0.0)
    overlap_area = torch.clamp(overlap_area, min=1e-15)  # Numerical stability
    repulsion = torch.sqrt(areai * areaj)
    penalty = repulsion * (overlap_area ** 2.5) * valid_mask.float()
    
    return penalty.sum()


# Example usage
if __name__ == "__main__":
    # Create test data
    N = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cell_features = torch.randn(N, 6, device=device, requires_grad=True)
    cell_features[:, 0] = torch.abs(cell_features[:, 0]) + 1.0  # area > 0
    cell_features[:, 4:6] = torch.abs(cell_features[:, 4:6]) + 1.0  # width, height > 0
    
    # Compute loss
    loss = overlap_loss_simplified(cell_features)
    print(f"Loss: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    print(f"Gradient norm: {cell_features.grad.norm().item():.6f}")
```

### 8.2 Toy CUDA Kernel

```cuda
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Simple overlap kernel for demonstration
__global__ void overlap_kernel_toy(
    const float* x,        // [N] cell x positions
    const float* y,        // [N] cell y positions
    const float* w,        // [N] cell widths
    const float* h,        // [N] cell heights
    int N,                 // Number of cells
    float* loss_out) {     // Output: scalar loss
    
    // Each thread processes one cell pair
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = N * (N - 1) / 2;
    
    if (idx >= total_pairs) return;
    
    // Convert linear index to (i, j) pair (upper triangle)
    // Formula: idx = i * (2*N - i - 1) / 2 + j - i - 1
    // Solve for i, j given idx
    int i = 0, j = 0;
    int remaining = idx;
    for (i = 0; i < N - 1; ++i) {
        int pairs_in_row = N - i - 1;
        if (remaining < pairs_in_row) {
            j = i + 1 + remaining;
            break;
        }
        remaining -= pairs_in_row;
    }
    
    // Load cell data
    float xi = x[i], yi = y[i];
    float xj = x[j], yj = y[j];
    float wi = w[i], hi = h[i];
    float wj = w[j], hj = h[j];
    
    // Compute overlap
    float dx = fabsf(xi - xj);
    float dy = fabsf(yi - yj);
    float min_sep_x = 0.5f * (wi + wj);
    float min_sep_y = 0.5f * (hi + hj);
    
    float overlap_x = fmaxf(0.0f, min_sep_x - dx);
    float overlap_y = fmaxf(0.0f, min_sep_y - dy);
    float overlap_area = overlap_x * overlap_y;
    
    // Simple penalty (no repulsion weight, no power law for simplicity)
    float penalty = overlap_area;
    
    // Atomic add to output
    if (penalty > 0.0f) {
        atomicAdd(loss_out, penalty);
    }
}

// Wrapper function
extern "C" void compute_overlap_toy(
    const float* x, const float* y, const float* w, const float* h,
    int N, float* loss_out) {
    
    int total_pairs = N * (N - 1) / 2;
    int threads_per_block = 256;
    int num_blocks = (total_pairs + threads_per_block - 1) / threads_per_block;
    
    overlap_kernel_toy<<<num_blocks, threads_per_block>>>(
        x, y, w, h, N, loss_out
    );
    
    cudaDeviceSynchronize();
}
```

### 8.3 Validation Example

```python
import torch
import numpy as np

def validate_overlap_loss():
    """Validate CUDA implementation against reference."""
    N = 1000
    device = torch.device("cuda")
    
    # Create test data
    cell_features = torch.randn(N, 6, device=device, requires_grad=True)
    cell_features[:, 0] = torch.abs(cell_features[:, 0]) + 1.0
    cell_features[:, 4:6] = torch.abs(cell_features[:, 4:6]) + 1.0
    
    # Reference implementation (PyTorch)
    loss_ref = overlap_loss_simplified(cell_features)
    loss_ref.backward()
    grad_ref = cell_features.grad.clone()
    
    # CUDA implementation (if available)
    try:
        from cuda_backend import compute_overlap_loss
        cell_features.grad = None
        loss_cuda = compute_overlap_loss(cell_features, ...)
        loss_cuda.backward()
        grad_cuda = cell_features.grad.clone()
        
        # Compare
        loss_diff = torch.abs(loss_ref - loss_cuda).item()
        grad_diff = torch.abs(grad_ref - grad_cuda).max().item()
        
        print(f"Loss difference: {loss_diff:.2e}")
        print(f"Gradient difference: {grad_diff:.2e}")
        
        assert loss_diff < 1e-4, "Loss mismatch"
        assert grad_diff < 1e-4, "Gradient mismatch"
        print("✓ Validation passed!")
    except ImportError:
        print("CUDA backend not available, skipping validation")
```

### 8.4 Performance Comparison

```python
import time
import torch

def benchmark_overlap_loss():
    """Benchmark PyTorch vs CUDA implementation."""
    N = 50000
    device = torch.device("cuda")
    cell_features = torch.randn(N, 6, device=device, requires_grad=True)
    cell_features[:, 0] = torch.abs(cell_features[:, 0]) + 1.0
    cell_features[:, 4:6] = torch.abs(cell_features[:, 4:6]) + 1.0
    
    # Warmup
    for _ in range(5):
        _ = overlap_loss_simplified(cell_features)
    
    # PyTorch timing
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        loss = overlap_loss_simplified(cell_features)
        loss.backward()
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / 10
    
    # CUDA timing (if available)
    try:
        from cuda_backend import compute_overlap_loss
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            loss = compute_overlap_loss(cell_features, ...)
            loss.backward()
        torch.cuda.synchronize()
        cuda_time = (time.time() - start) / 10
        
        speedup = pytorch_time / cuda_time
        print(f"PyTorch: {pytorch_time:.3f}s")
        print(f"CUDA: {cuda_time:.3f}s")
        print(f"Speedup: {speedup:.1f}x")
    except ImportError:
        print(f"PyTorch: {pytorch_time:.3f}s")
        print("CUDA backend not available")
```

---

## Conclusion

This document provides a comprehensive breakdown of the overlap loss algorithm, from mathematical formulation to CUDA implementation. Key takeaways:

1. **Mathematical foundation**: The loss uses a super-quadratic penalty with adaptive margin and smooth surrogate gradients.

2. **Optimization strategies**: Spatial hashing reduces complexity from O(N²) to O(N), while chunking manages GPU memory.

3. **CUDA implementation**: Grid-stride loops, shared memory tiling, and careful synchronization enable efficient GPU execution.

4. **Best practices**: Validate correctness first, then optimize. Use profiling tools to identify bottlenecks.

5. **Architecture**: Separate files for kernels, wrappers, and Python interface improve maintainability and compilation speed.

For questions or improvements, refer to the codebase or create an issue.

