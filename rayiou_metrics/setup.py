from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import torch

ext_modules = []

if torch.cuda.is_available():
    ext_modules.append(
        CUDAExtension(
            name='dvr_cuda_ops',
            sources=[
                'lib/dvr_cuda/dvr_cuda.cpp',
                'lib/dvr_cuda/dvr_render_cuda.cu'
            ],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-allow-unsupported-compiler']}
        )
    )

ext_modules.append(
    CppExtension(
        name='dvr_cpu_ops',
        sources=[
            'lib/dvr_cpu/dvr.cpp',
            'lib/dvr_cpu/dvr_render_cpu.cpp'
        ]
    )
)

setup(
    name='rayiou_metrics',
    version='1.0',
    description='A package for calculating ray metrics with CPU and CUDA support.',
    packages=find_packages(),
    package_data={
        'rayiou_metrics': ['ego_infos_val.pkl']  # Include .pkl in the package
    },
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch',
        'numpy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
