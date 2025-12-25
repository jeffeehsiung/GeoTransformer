# setup.py
import os

from setuptools import find_packages, setup

# Try to import torch's extension helpers, but don't hard-fail if torch isn't installed yet
try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa

    _HAVE_TORCH_EXT = True
except Exception:
    BuildExtension = None
    CUDAExtension = None
    _HAVE_TORCH_EXT = False


def read_requirements():
    reqs = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            reqs = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return reqs


def make_ext_modules():
    """Return ext_modules and cmdclass, possibly empty if torch/cuda not available or disabled."""
    if not _HAVE_TORCH_EXT:
        return [], {}

    # Allow disabling native build during dev/CI:
    if os.environ.get("GEO_BUILD_EXT", "1") != "1":
        return [], {}

    from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # lazy import

    ext_modules = [
        CUDAExtension(
            name="geotransformer.ext",
            sources=[
                "geotransformer/extensions/extra/cloud/cloud.cpp",
                "geotransformer/extensions/cpu/grid_subsampling/grid_subsampling.cpp",
                "geotransformer/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp",
                "geotransformer/extensions/cpu/radius_neighbors/radius_neighbors.cpp",
                "geotransformer/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp",
                "geotransformer/extensions/pybind.cpp",
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
    ]
    return ext_modules, {"build_ext": BuildExtension}


ext_modules, cmdclass = make_ext_modules()

setup(
    name="geotransformer",
    version="1.0.0",
    description="GeoTransformer: Fast and Robust Point Cloud Registration with Geometric Transformer",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["*.py", "*.txt", "*.md", "*.yml", "*.yaml", "*.json"],
        "geotransformer": [
            "**/*.py",
            "**/*.txt",
            "**/*.md",
            "**/*.yml",
            "**/*.yaml",
            "**/*.json",
            "**/*.so",
            "extensions/**/*.cpp",
            "extensions/**/*.cu",
            "extensions/**/*.h",
            "extensions/**/*.hpp",
        ],
        "experiments": ["**/*.py"],
        "data": ["**/*.py", "**/*.txt", "**/*.md", "**/*.yml", "**/*.yaml", "**/*.json"],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=[
        "numpy",
        "torch>=1.6.0",  # kept as a runtime dependency; setup can parse without torch present
        "open3d>=0.13.0",
        "tqdm",
        "easydict",
        "opencv-python",
        "scipy",
        "matplotlib",
        "plyfile",
        "scikit-learn",
        "nibabel",
    ]
    + read_requirements(),
    python_requires=">=3.6",
    zip_safe=False,
)
