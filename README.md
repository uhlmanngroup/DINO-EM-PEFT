# Microscopy Foundation Hub
[![arXiv](https://img.shields.io/badge/arXiv-2602.08505-b31b1b.svg)](https://arxiv.org/abs/2602.08505)

A foundation‑model hub for microscopy: training, fine‑tuning, and evaluation across modalities.
More details to follow.

**Quickstart**
1. Create an environment and install the package (local).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

2. Pick a config and train.

```bash
python scripts/train_em_seg.py --cfg configs/mac/lucchi_dinov2_lora_mac.yaml
```

3. (Optional) evaluate and summarize.

```bash
python scripts/eval_em_seg.py --cfg configs/mac/lucchi_dinov2_lora_mac.yaml
python scripts/analysis/summarize_seg_results.py --root /path/to/results/seg
```

**Configs And Paths**
- All parameters live in YAML under `configs/`. CLI flags and sbatch env vars only override config values.
- Many configs contain absolute paths from the author's machines (for example `/Users/cfuste/...` or `/home/cfuste/...`). Keep them if you use the same layout, otherwise update them for your environment.
- See `configs/README.md` for the full configuration map and environment YAMLs.

**Common Workflows**
- Train: `python scripts/train_em_seg.py --cfg configs/mac/...yaml`
- Eval: `python scripts/eval_em_seg.py --cfg configs/mac/...yaml`
- Summaries: `python scripts/analysis/summarize_seg_results.py --root /path/to/results/seg`
- Summary plots: `python scripts/analysis/plot_seg_summary.py --summary-dir /path/to/results/seg/summary`
- Feature extraction: `python scripts/extract_features.py --cfg configs/mac/em_unsupervised_features_mac.yaml`
- PCA/UMAP: `python scripts/run_pca.py --cfg configs/mac/em_pca_mac.yaml`
- OOD detection: `python scripts/ood_detection.py --cfg configs/mac/ood_detection_mac.yaml`
- Domain analysis: `python scripts/run_domain_analysis.py --cfg configs/mac/domain_analysis.yaml`
- Backbone inspection: `python scripts/inspect_backbone.py --cfg configs/mac/paired_openclip_vitl14.yaml`

For a full list of entrypoints and their purpose, see `scripts/README.md`.

**Cluster (SLURM)**
- Train + eval (single run): `sbatch slurm/single_lucchi_dinov2.sbatch configs/cluster/lucchi_dinov2_cluster.yaml`
- Feature grids: `sbatch slurm/feat_analysis_paired.sbatch`
- Summaries: `sbatch slurm/summarize_seg_results.sbatch`
- Param counts: `sbatch slurm/param_counts.sbatch`

**Backbone Selection**
- Backbone choice is config-driven via a `backbone` block. See `configs/backbones/` for examples.
- Legacy configs that only specify `dino_size` still work and default to DINOv2.

**DINOv3 Weights**

Access to DINOv3 weights requires an approved request. Once you receive the link, download the checkpoint and set `backbone.weights` and `backbone.repo_dir` in your config. The paths below are placeholders:

```bash
wget -O /path/to/dinov3_vits16_pretrain.pth "<DINOv3_CHECKPOINT_URL>"
```

```yaml
backbone:
  name: dinov3
  variant: vits16
  repo_dir: /path/to/facebookresearch/dinov3
  weights: /path/to/dinov3_vits16_pretrain.pth
```

Also update `img_size.patch_multiple` to 16 when using DINOv3 so resizing snaps to the correct stride.

**OpenCLIP Weights And Caching**

Use `backbone.pretrained` to point to an OpenCLIP tag (auto-downloads if cached access is available) or set `backbone.weights` to a local checkpoint path for offline clusters. Cache locations can be steered via `HF_HOME` or `XDG_CACHE_HOME`.

**Results Layout**

All runs share a common results layout driven by the YAML config. Each config must define:

- `experiment_id`: unique run name (e.g. `2025-11-20_A1_lucchi+droso_dinov2-base_lora-none_seg`)
- `results_root`: root directory for all outputs
- `task_type`: `seg`, `feats`, `ood-detection`, etc.

Outputs land at:

```
<results_root>/<task_type>/<experiment_id>/
  config_used.yaml
  run_info.txt
  metrics.json
  ckpts/
  figs/
```

`metrics.json` is updated by training/eval/analysis scripts via `update_metrics()`.

**Repository Structure**

```
shift-peft/
  configs/
    README.md
    backbones/           # Backbone config examples
    cluster/             # Cluster configs
    environments/        # Minimal conda env YAMLs
    mac/                 # Local configs
  docs/
    media/               # README assets
  scripts/
    README.md
    analysis/            # summarize/plot/fancy/OOD comparison
    data/                # dataset composition utilities
    utils/               # log helpers + param counts
    train_em_seg.py
    eval_em_seg.py
    extract_features.py
    run_pca.py
    run_domain_analysis.py
    ood_detection.py
    smoke_test.py
    smoke_test_lora.py
    inspect_backbone.py
    ablation_paired_balance_dinov2.py
  slurm/                 # sbatch entrypoints
  src/                   # core library code
  README.md
  pyproject.toml
```

**Datasets**

Lucchi, A., Smith, K., Achanta, R., Knott, G., & Fua, P. (2011). Supervoxel-based segmentation of mitochondria in em image stacks with learned shape features. IEEE transactions on medical imaging, 31(2), 474-486. Download [here](https://casser.io/connectomics).

Casser, V., Kang, K., Pfister, H., & Haehn, D. (2020, September). Fast mitochondria detection for connectomics. In Medical Imaging with Deep Learning (pp. 111-120). PMLR. Download [here](https://github.com/unidesigner/groundtruth-drosophila-vnc/tree/master).

For usability purposes, the two datasets are composed into:
```bash
<BASE>/composed-dinopeft/
  train/images, train/masks
  test/images,  test/masks
  mapping.csv
```

To build that layout, edit paths in `scripts/data/compose_em_datasets.py` and run:
```bash
python scripts/data/compose_em_datasets.py
```

**Cite**

If you use this work, cite the arXiv preprint [arXiv:2602.08505](https://arxiv.org/abs/2602.08505). The BibTeX entry is available on the arXiv page.

**Acknowledgements**

This project stands on the shoulders of excellent open-source work and research. We’re grateful to the authors and maintainers of the following projects and papers:

- **DINOv2 (Meta AI / Facebook Research)**
  We use DINOv2 Vision Transformers and public pretraining weights (loaded via `torch.hub`) as our frozen backbone. DINOv2 provides strong, general-purpose visual representations that we adapt to electron microscopy via parameter-efficient fine-tuning (PEFT).
  Repo: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

- **FD-DINOv2 (Fréchet Distance with DINOv2 backbone)**
  We adapt the idea of replacing Inception features in FID with DINOv2 features, as implemented in the FD-DINOv2 repository. In our case, we compute a FID-style Fréchet distance between Gaussian approximations of the DINOv2 feature distributions of two EM datasets (with and without LoRA) to quantify domain shift.
  Repo: [justin4ai/FD-DINOv2](https://github.com/justin4ai/FD-DINOv2)

- **Exposing Flaws of Generative Model Evaluation Metrics**
  Stein et al. systematically study generative model evaluation metrics and show that relying on Inception-V3 can unfairly penalize diffusion models. They propose using self-supervised encoders such as DINOv2-ViT-L/14 as richer feature extractors for Fréchet-based scores, which motivates our choice to replace Inception with DINOv2 embeddings when measuring domain distances.
  Paper: *Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models*, NeurIPS 2023.

- **DINOSim (Electron Microscopy zero-shot evaluation)**
  DINOSim motivates our evaluation focus: it investigates zero-shot detection/segmentation on EM imagery using DINO features and highlights the domain gap for microscopy. We build on that insight by demonstrating how PEFT (LoRA) improves downstream EM segmentation compared to zero-shot.
  Project/Paper: *DINOSim: Zero-Shot Object Detection and Semantic Segmentation on Electron Microscopy Images.*

**Licensing note:**
Please review and respect the licenses of upstream repositories (e.g. DINOv2) and any datasets you use. Their terms apply to model weights, code, and data redistributed or fine-tuned within this project.
