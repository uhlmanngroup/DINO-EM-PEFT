# src/dino_peft/analysis/feature_extractor.py
# Python file for extracting features from images using DINO backbones.

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from dino_peft.backbones import (
    build_backbone,
    build_preprocess_transform,
    resolve_backbone_cfg,
    resolve_preprocess_cfg,
)
from dino_peft.datasets.flat_image_folder import FlatImageFolder
from dino_peft.models.lora import apply_peft

def _count_lora_matches(lora_state: dict, model_state: dict) -> int:
    return sum(1 for k in lora_state.keys() if k in model_state)


def _remap_lora_state(lora_state: dict, model_state: dict) -> tuple[dict, str]:
    if not lora_state:
        return lora_state, "empty"

    candidates = []
    base_matches = _count_lora_matches(lora_state, model_state)
    candidates.append(("as-is", lora_state, base_matches))

    if any(k.startswith("vit.") for k in lora_state.keys()):
        stripped = {k[len("vit."):]: v for k, v in lora_state.items()}
        candidates.append(("strip-vit", stripped, _count_lora_matches(stripped, model_state)))

    if any(k.startswith("vit.") for k in model_state.keys()):
        prefixed = {
            (k if k.startswith("vit.") else f"vit.{k}"): v
            for k, v in lora_state.items()
        }
        candidates.append(("add-vit", prefixed, _count_lora_matches(prefixed, model_state)))

    best = max(candidates, key=lambda item: item[2])
    return best[1], best[0]

@torch.no_grad()
def extract_features_from_folder(
    data_dir: str | Path,
    dino_size: str = "base",
    img_size: int | dict | tuple = 518,
    batch_size: int = 16,
    num_workers: int = 4,
    device: str = "cuda",
    checkpoint_path: str | Path | None = None,
    backbone_cfg: dict | None = None,
):
    """
    Run the configured DINO backbone on all images in a folder and return .npz features.

    Assumptions:
    - Images are already in the size/formad used for EM segmentation.
    - em_dino_unsup_transforms applies ToTensor and ImageNet normalization.

    Args:
        data_dir: path to folder with images.
        dino_size: legacy DINOv2 size (e.g., "base"); ignored if backbone_cfg provided.
        batch_size: dataloader batch size.
        num_workers: dataloader num_workers.
        device: device to run model on.

    Returns:
        features:            [N, C] float32 (global-average-pooled over H', W')
        image_paths:         [N] object array of strings (paths to images)
        dataset_ids:         [N] int32 (0=lucchi, 1=droso)
        dataset_names:       [N] object array of strings ("lucchi" or "droso")
        dataset_name_to_id:  [K] object array of "name:id" strings
        size:                [1] object array size of the DINO model, e.g. ["base"]
        checkpoint_path: optional checkpoint containing LoRA weights to load.
    """
    data_dir = Path(data_dir)
    checkpoint_path = Path(checkpoint_path).expanduser() if checkpoint_path else None
    device_obj = device if isinstance(device, torch.device) else torch.device(device)
    device_type = device_obj.type

    if backbone_cfg is None:
        backbone_cfg = resolve_backbone_cfg(
            {
                "dino_size": dino_size,
            }
        )
    model = build_backbone(backbone_cfg, device=device_obj)
    model.eval()

    if checkpoint_path:
        print(f"[feature_extractor] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device_obj)
        ckpt_cfg = ckpt.get("cfg", {}) or {}
        ckpt_backbone = resolve_backbone_cfg(ckpt_cfg)
        ckpt_variant = ckpt_backbone.get("variant")
        if ckpt_variant and ckpt_variant != backbone_cfg.get("variant"):
            print(
                "[feature_extractor] WARNING: checkpoint backbone "
                f"{ckpt_backbone.get('name')}:{ckpt_variant} != requested "
                f"{backbone_cfg.get('name')}:{backbone_cfg.get('variant')}"
            )
        audit = apply_peft(
            model.model,
            ckpt_cfg,
            backbone_info=ckpt_backbone,
            write_report=False,
        )
        if audit is not None:
            replaced = audit.targets
            lora_state = ckpt.get("backbone_lora") or {}
            if not lora_state:
                raise RuntimeError("Checkpoint does not contain backbone_lora weights.")
            model_state = model.model.state_dict()
            lora_state, remap_strategy = _remap_lora_state(lora_state, model_state)
            if remap_strategy != "as-is":
                print(f"[feature_extractor] Remapped LoRA keys ({remap_strategy}).")
            missing = []
            for key, tensor in lora_state.items():
                if key in model_state:
                    model_state[key] = tensor
                else:
                    missing.append(key)
            if missing:
                raise RuntimeError(
                    f"Missing LoRA keys in backbone: {missing[:5]}{'...' if len(missing) > 5 else ''}"
                )
            model.model.load_state_dict(model_state)
            model.eval()
            print(f"[feature_extractor] Loaded LoRA weights ({len(replaced)} layers) from checkpoint.")
        else:
            print("[feature_extractor] Checkpoint has no LoRA weights to load; using base backbone.")

    # Data loader using transforms for EM segmentation
    preprocess_cfg = resolve_preprocess_cfg({"backbone": backbone_cfg}, default_img_size=img_size)
    transform = build_preprocess_transform(
        preprocess_cfg["preset"],
        preprocess_cfg["img_size"],
        backbone_cfg=backbone_cfg,
    )
    dataset = FlatImageFolder(root_dir=data_dir, transform=transform)

    def pad_collate(batch):
        images, paths = zip(*batch)
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        padded = []
        for img in images:
            pad_h = max_h - img.shape[1]
            pad_w = max_w - img.shape[2]
            padded.append(F.pad(img, (0, pad_w, 0, pad_h)))
        return torch.stack(padded, dim=0), list(paths)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device_type == "cuda"),
        collate_fn=pad_collate,
    )

    all_features = []
    all_paths = []
    all_dataset_names = []

    for imgs, paths in loader:
        imgs = imgs.to(device_obj, non_blocking=True)
        output = model(imgs)
        feats_np = output.global_embedding.cpu().numpy().astype("float32")
        all_features.append(feats_np)
        all_paths.extend(paths)
        # Infer dataset name from filename prefix
        for p in paths:
            name = Path(p).name
            if name.startswith("lucchi"):
                all_dataset_names.append("lucchi")
            elif name.startswith("droso"):
                all_dataset_names.append("droso")
            else:
                raise ValueError(f"Cannot infer dataset name from filename '{name}'")

    features_np = np.concatenate(all_features, axis=0)  # (N, C)
    unique_names = sorted(set(all_dataset_names))
    name_to_id = {name: idx for idx, name in enumerate(unique_names)}
    dataset_ids = np.array([name_to_id[n] for n in all_dataset_names], dtype=np.int32)
    dataset_name_to_id = np.array([f"{name}:{idx}" for name, idx in name_to_id.items()], dtype=object)

    return {
        "features": features_np,
        "image_paths": np.array(all_paths, dtype=object),
        "dataset_ids": dataset_ids,
        "dataset_names": np.array(all_dataset_names, dtype=object),
        "dataset_name_to_id": dataset_name_to_id,
        "dino_size": np.array([backbone_cfg["variant"]], dtype=object),
        "backbone_name": np.array([backbone_cfg["name"]], dtype=object),
        "backbone_variant": np.array([backbone_cfg["variant"]], dtype=object),
    }
