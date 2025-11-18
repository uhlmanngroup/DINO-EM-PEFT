# src/dino_peft/analysis/feature_extractor.py
# Python file for extracting features from images using DINOv2 backbone model.

import torch
import numpy as np
from pathlib import Path

from dino_peft.utils.transforms import em_dino_unsup_transforms
from dino_peft.datasets.flat_image_folder import FlatImageFolder
from dino_peft.models.backbone_dinov2 import DINOv2FeatureExtractor

@torch.no_grad()
def extract_features_from_folder(
    data_dir: str | Path,
    dino_size: str = "base",
    img_size: int = 518, 
    batch_size: int = 16,
    num_workers: int = 4,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Run DINOv2 feature extractor on all images in a folder and return .npz file with features.

    Assumptions:
    - Images are already in the size/formad used for EM segmentation.
    - em_seg_transforms applies ToTensor and ImageNet normalization.

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
        dataset_name_to_id:  [K]    object array of "name:id" strings
        size:                [1]    object array size of the DINO model, e.g. ["base"]
    """
    data_dir = Path(data_dir)

    # Build model 
    model = DINOv2FeatureExtractor(size=dino_size, device=device)
    model.eval() 

    # Data loader using transforms for EM segmentation
    transform = em_dino_unsup_transforms(img_size=img_size)
    dataset = FlatImageFolder(root_dir=data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    all_features = []
    all_paths = []
    all_dataset_names = []

    for imgs, paths in loader:
        imgs = imgs.to(device, non_blocking=True)
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
            
        # Map dataset names to IDs
        unique_names = sorted(set(all_dataset_names))
        name_to_id = {name: idx for idx, name in enumerate(unique_names)}
        dataset_ids = np.array([name_to_id[n] for n in all_dataset_names], dtype=np.int32)

        # Store mapping as "name:id" strings for easy reconstruction later
        dataset_name_to_id = np.array(
            [f"{name}:{idx}" for name, idx in name_to_id.items()], dtype=object
        )  

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


        
                
