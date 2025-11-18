# src/dino_peft/analysis/dimred.py

from pathlib import Path
from typing import Tuple

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

        return FeatureBundle(
            features=features,
            dataset_ids=dataset_ids,
            dataset_names=dataset_names,
            image_paths=image_paths,
            meta={
                "dataset_name_to_id": name_to_id,
                "raw_keys": list(data.keys()),
                "source": str(path),
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
