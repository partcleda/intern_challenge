"""Build script for CUDA overlap computation extension.

Compiles overlap_cuda.cpp and overlap_cuda_kernel.cu into a PyTorch extension module.
"""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()

setup(
    name="overlap_cuda_backend",
    ext_modules=[
        CUDAExtension(
            name="overlap_cuda_backend",
            sources=[
                str(script_dir / "overlap_cuda.cpp"),
                str(script_dir / "overlap_cuda_kernel.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
