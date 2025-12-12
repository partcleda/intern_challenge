"""
CUDA backend setup and checking.
"""

import torch

# Import CUDA_OVERLAP_AVAILABLE from losses module
# This is defined when losses.py is imported
try:
    from placement_modules.losses import CUDA_OVERLAP_AVAILABLE
except ImportError:
    # If losses hasn't been imported yet, set default
    CUDA_OVERLAP_AVAILABLE = False

def check_and_setup_cuda_backend():
    """Check CUDA availability and inform user about GPU acceleration.
    
    Returns:
        bool: True (always continue - PyTorch handles GPU automatically)
    """
    # Import here to ensure losses module is loaded
    from placement_modules.losses import CUDA_OVERLAP_AVAILABLE
    
    # Check if CUDA is available on the system
    cuda_available = torch.cuda.is_available()
    
    if not cuda_available:
        print("\n" + "=" * 70)
        print("CUDA NOT AVAILABLE")
        print("=" * 70)
        print("CUDA is not available on this system.")
        print("The code will run using the CPU-optimized PyTorch implementation.")
        print("This is slower but will still work correctly.\n")
        return True
    
    # CUDA is available - check if custom backend is available
    if not CUDA_OVERLAP_AVAILABLE:
        print("\n" + "=" * 70)
        print("GPU ACCELERATION INFO")
        print("=" * 70)
        print("CUDA is available on this system.")
        print("Custom CUDA backend is not available (not built or import failed).")
        print("PyTorch operations will automatically use GPU acceleration")
        print("when tensors are on GPU device.\n")
        return True
    
    # Custom backend is available - it will be used for large problems (50k+ cells)
    print("\n" + "=" * 70)
    print("CUDA BACKEND AVAILABLE")
    print("=" * 70)
    print("Custom CUDA backend is available and will be used for problems")
    print("with 50,000+ cells. This provides 2-3x speedup over PyTorch.\n")
    return True
    

