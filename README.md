# Robust Derivative Estimation with Walk on Stars

This repository contains the implementation for the paper:

**Robust Derivative Estimation with Walk on Stars**  
Zihan Yu, Rohan Sawhney, Bailey Miller, Lifan Wu, Shuang Zhao

[[Project Page]](https://projects.shuangz.com/grad-wost-sa25/) [[Paper]](https://projects.shuangz.com/grad-wost-sa25/paper.pdf)

## Overview

This implementation uses a hybrid C++/Python architecture. The BVH (Bounding Volume Hierarchy) and SNCH (Spatial Normal Cone Hierarchy) data structures are constructed in C++ (adapted from [FCPW](https://github.com/rohan-sawhney/fcpw)) and exposed to Python via nanobind. Geometry traversal and Monte Carlo solvers are implemented in Python with [Dr.Jit](https://github.com/mitsuba-renderer/drjit) for GPU-accelerated parallel evaluation.

```
diff_wost/      # Geometry queries and solvers (GPU-accelerated via Dr.Jit)
include/ & src/ # C++ extensions for BVH and SNCH construction (adapted from FCPW)
examples/       # Example scripts for 2D and 3D gradient estimation
assets/         # Sample meshes (OBJ files)
```

## Installation

Requires Python 3.9+ and a C++ compiler. The package includes C++ extensions that will be built during installation.

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt install libeigen3-dev cmake

# Install package
pip install -e .
``` 

## Examples

```bash
python examples/example_2d/run.py
python examples/example_3d/run.py
```

Output files will be saved to the `out/` folder:

- **2D example**: Generates streamline visualization as PNG/PDF images and raw data in `.npy` format (NumPy array).
- **3D example**: Outputs streamline geometry as `.obj` file, which can be imported into Blender or other 3D software for visualization.

## Citation

```bibtex
@article{Yu:2025:GradWost,
    author = {Yu, Zihan and Sawhney, Rohan and Miller, Bailey and Wu, Lifan and Zhao, Shuang},
    title = {Robust Derivative Estimation with Walk on Stars},
    journal = {ACM Trans. Graph.},
    volume = {44},
    number = {6},
    year = {2025},
    pages = {253:1--253:16}
}
```

## License

MIT
