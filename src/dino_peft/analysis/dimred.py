# src/dino_peft/analysis/dimred.py

from pathlib import Path
from typing import Tuple

import umap
import numpy as np
from sklearn.decomposition import PCA

class FeatureBundle:
    def __init__(self, features, dataset_ids=None, dataset_names=None, image_paths=None, meta=None):
        self.features = features
        self.dataset_ids = dataset_ids
        self.dataset_names = dataset_names
        self.image_paths = image_paths
        self.meta = meta or {} # save additional features

def load_feature_npz(path: Path | str) -> FeatureBundle:
    """Load a .npz produced by extract_features, keeping paths/labels aligned."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        features = np.asarray(data["features"])
        dataset_ids = data.get("dataset_ids")
        dataset_names = data.get("dataset_names")
        image_paths = data.get("image_paths")
        dataset_name_to_id = data.get("dataset_name_to_id")
        dino_size = data.get("dino_size")
        backbone_name = data.get("backbone_name")
        backbone_variant = data.get("backbone_variant")
        
        if dino_size is not None:
            try:
                dino_size = dino_size.tolist()[0]
            except Exception:
                pass  # leave as-is

        # Normalize dtypes
        if dataset_ids is not None:
            dataset_ids = np.asarray(dataset_ids).astype(np.int64, copy=False)
        if dataset_names is not None:
            dataset_names = list(dataset_names.tolist())
        if image_paths is not None:
            image_paths = list(image_paths.tolist())

        # Rebuild label mapping if possible
        name_to_id = None
        if dataset_name_to_id is not None:
            name_to_id = {
                name: int(idx)
                for name, idx in (
                    s.split(":") for s in dataset_name_to_id.tolist()
                )
            }

        def _to_scalar(value):
            if value is None:
                return None
            try:
                value = value.tolist()
            except Exception:
                return value
            if isinstance(value, (list, tuple)) and value:
                return value[0]
            return value

        dino_size = _to_scalar(dino_size)
        backbone_name = _to_scalar(backbone_name)
        backbone_variant = _to_scalar(backbone_variant)

        return FeatureBundle(
            features=features,
            dataset_ids=dataset_ids,
            dataset_names=dataset_names,
            image_paths=image_paths,
            meta={
                "dataset_name_to_id": name_to_id,
                "raw_keys": list(data.keys()),
                "source": str(path),
                "dino_size": dino_size,
                "backbone_name": backbone_name,
                "backbone_variant": backbone_variant,
            },
        )

def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2 normalize (useful for cosine-ish geometry)."""
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom

def run_pca(
    features: np.ndarray,
    n_components: int = 2,
    whiten: bool = False,
    random_state: int = 0,
    l2norm: bool = False,
) -> Tuple[PCA, np.ndarray]:
    """
    Fit PCA and return the fitted object + transformed embeddings.

    - l2norm: normalize rows before PCA (optional).
    - Raises ValueError if n_components > min(N, D).
    """
    x = np.asarray(features)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    n, d = x.shape
    if n_components > min(n, d):
        raise ValueError(
            f"n_components={n_components} must be <= min(n_samples={n}, n_features={d})"
        )
    if l2norm:
        x = l2_normalize(x)

    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    emb = pca.fit_transform(x)
    return pca, emb

def run_umap(features, n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean", random_state=0):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    )
    emb = reducer.fit_transform(features)
    return reducer, emb
