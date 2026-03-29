## Installation

### Requirements
- Linux or macOS with Python ≥ 3.10.
- PyTorch 2.x and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenCV is optional but needed by demo and visualization.

Example setup (CPU-first):

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

```bash
cd third_party/Mask2Former
python setup.py build install
```

#### Building on another system
To build on a system that does not have a GPU device but provide the drivers:
```bash
TORCH_CUDA_ARCH_LIST='8.0' FORCE_CUDA=1 python setup.py build install
```

### Example environment setup
```bash
cd third_party/Mask2Former
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
python setup.py build install
```

To keep your path aligned with CPU-first workflows used in this fork, install CPU wheels first:

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -e .
```
