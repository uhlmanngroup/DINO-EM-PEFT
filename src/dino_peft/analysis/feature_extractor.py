# src/dino_peft/analysis/feature_extractor.py
# Python file for extracting features from images using DINOv2 backbone model.

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from dino_peft.utils.transforms import em_dino_unsup_transforms
from dino_peft.datasets.flat_image_folder import FlatImageFolder
from dino_peft.models.backbone_dinov2 import DINOv2FeatureExtractor
from dino_peft.models.lora import inject_lora

@torch.no_grad()
def extract_features_from_folder(
    data_dir: str | Path,
    dino_size: str = "base",
    img_size: int | dict | tuple = 518,
    batch_size: int = 16,
    num_workers: int = 4,
    device: str = "cuda",
    checkpoint_path: str | Path | None = None,
):
    """
    Run DINOv2 feature extractor on all images in a folder and return .npz file with features.

    Assumptions:
    - Images are already in the size/formad used for EM segmentation.
    - em_dino_unsup_transforms applies ToTensor and ImageNet normalization.

    Args:
        data_dir: path to folder with images.
        dino_size: which DINOv2 model size to use ("base", "large", etc).
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

    # Build model 
    model = DINOv2FeatureExtractor(size=dino_size, device=device_obj)
    model.eval()

    if checkpoint_path:
        print(f"[feature_extractor] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device_obj)
        ckpt_cfg = ckpt.get("cfg", {}) or {}
        ckpt_dino = ckpt_cfg.get("dino_size")
        if ckpt_dino and ckpt_dino != dino_size:
            print(f"[feature_extractor] WARNING: checkpoint dino_size={ckpt_dino} != requested {dino_size}")
        use_lora = bool(ckpt_cfg.get("use_lora", False))
        lora_rank = int(ckpt_cfg.get("lora_rank", 0) or 0)
        lora_alpha = int(ckpt_cfg.get("lora_alpha", 0) or 0)
        lora_targets = ckpt_cfg.get("lora_targets", ["attn.qkv", "attn.proj"])
        if use_lora and lora_rank > 0:
            replaced = inject_lora(
                model.vit,
                target_substrings=lora_targets,
                r=lora_rank,
                alpha=lora_alpha if lora_alpha > 0 else lora_rank,
            )
            lora_state = ckpt.get("backbone_lora") or {}
            if not lora_state:
                raise RuntimeError("Checkpoint does not contain backbone_lora weights.")
            model_state = model.state_dict()
            missing = []
            for key, tensor in lora_state.items():
                if key in model_state:
                    model_state[key] = tensor
                else:
                    missing.append(key)
            if missing:
                raise RuntimeError(f"Missing LoRA keys in backbone: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            model.load_state_dict(model_state)
            model.eval()
            print(f"[feature_extractor] Loaded LoRA weights ({len(replaced)} layers) from checkpoint.")
        else:
            print("[feature_extractor] Checkpoint has no LoRA weights to load; using base backbone.")

    # Data loader using transforms for EM segmentation
    transform = em_dino_unsup_transforms(img_size=img_size)
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
        features = model(imgs)  # (B, C, H', W')
        features = features.mean(dim=[2, 3])  # global average pool to (B, C)
        feats_np = features.cpu().numpy().astype("float32")
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
    "dino_size": np.array([dino_size], dtype=object),
}


        
                
