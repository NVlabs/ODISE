# ODISE — Open-Vocabulary Panoptic Segmentation

Open-vocabulary panoptic segmentation using pre-trained text-image diffusion and discriminative models (CVPR 2023 Highlight, NVIDIA).

## Architecture
```
odise/
├── checkpoint/     # Custom checkpointer (ODISE weights)
├── config/         # Detectron2-style configs
├── data/           # Dataset registration & transforms
├── engine/         # Training loop & defaults
├── evaluation/     # Eval metrics
├── model_zoo/      # Pre-built model configs
├── modeling/       # Core models (diffusion, meta-arch, backbone, wrapper)
└── utils/          # Env collection, misc helpers
configs/            # YAML/Python training configs
third_party/        # Mask2Former, latent-diffusion, taming-transformers
tools/              # train_net.py, extract_features.py, bootstrap script
demo/               # Gradio demo app
```

## Key Dependencies
- Python >=3.10, PyTorch >=2.0
- detectron2, Mask2Former (local third_party)
- open-clip-torch==2.0.2, timm==0.6.11
- numpy<2.0, omegaconf>=2.3
- Stable Diffusion via latent-diffusion/taming-transformers submodules

## Dev Commands
```bash
# Activate env (GPU server)
source /mnt/forge-data/activate.sh

# Install
uv pip install -e .

# Bootstrap third-party submodules
bash tools/bootstrap_third_party.sh

# Train
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file configs/common/train.py --num-gpus 4

# Demo
python demo/demo.py

# Lint
ruff check odise/ --select E,F,I,B,UP
isort --check odise/
mypy odise/
```

## Conventions
- Package manager: `uv` (never pip directly)
- Search: `rg` (ripgrep), never `grep`
- Line length: 100
- Style: isort + ruff
- Config: Detectron2 LazyConfig system (Python-based configs)
- Git commit prefix: `[ODISE]`
- Training outputs: `/mnt/artifacts-datai/`

# currentDate
Today's date is 2026-03-29.
