from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="overlap_cuda_backend",
    ext_modules=[
        CUDAExtension(
            name="overlap_cuda_backend",
            sources=[
                "cuda_backend/overlap_cuda.cpp",
                "cuda_backend/overlap_cuda_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
