from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .base import (
    BackboneAdapter,
    BackboneOutput,
    infer_grid_size,
    infer_num_register_tokens,
    infer_patch_size,
    split_special_tokens,
)


def _resolve_model_name(variant: str) -> str:
    if variant.startswith("dinov3_"):
        return variant
    return f"dinov3_{variant}"


def _load_state_dict(weights: str, device: torch.device) -> dict:
    weights_path = Path(weights).expanduser()
    if weights_path.exists():
        state = torch.load(weights_path, map_location=device)
    else:
        if not str(weights).startswith(("http://", "https://")):
            raise FileNotFoundError(f"DINOv3 weights not found: {weights}")
        state = torch.hub.load_state_dict_from_url(weights, map_location=device)
    if isinstance(state, dict):
        if "model" in state:
            return state["model"]
        if "state_dict" in state:
            return state["state_dict"]
    return state


class DINOv3Adapter(BackboneAdapter):
    def __init__(
        self,
        variant: str,
        device: str | torch.device,
        *,
        load_backend: str = "torchhub",
        repo_dir: str | None = None,
        weights: str | None = None,
        num_register_tokens: int | None = None,
    ) -> None:
        super().__init__(name="dinov3", variant=variant)
        if load_backend != "torchhub":
            raise ValueError(f"Unsupported load_backend '{load_backend}'.")

        repo = repo_dir
        if repo_dir:
            repo_path = Path(repo_dir).expanduser()
            if not repo_path.exists():
                raise FileNotFoundError(f"DINOv3 repo_dir not found: {repo_path}")
            repo = str(repo_path)
        repo = repo or "facebookresearch/dinov3"
        source = "local" if repo_dir else "github"
        model_name = _resolve_model_name(variant)
        model = torch.hub.load(repo, model_name, source=source, pretrained=False)
        model.eval()

        if not weights:
            raise ValueError(
                "DINOv3 weights are required. Set backbone.weights to a local path or URL."
            )
        state = _load_state_dict(weights, device=torch.device(device))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[dinov3] Missing keys when loading weights: {missing[:5]}")
        if unexpected:
            print(f"[dinov3] Unexpected keys when loading weights: {unexpected[:5]}")

        self.model = model
        self.embed_dim = getattr(model, "embed_dim", 0)
        if not self.embed_dim:
            raise ValueError("Unable to infer embed_dim from DINOv3 model.")
        self.patch_size = infer_patch_size(model, fallback=16)
        if self.patch_size <= 0:
            raise ValueError("Unable to infer patch_size from DINOv3 model.")
        self.num_register_tokens = (
            int(num_register_tokens)
            if num_register_tokens is not None
            else infer_num_register_tokens(model)
        )
        self.to(device)
        self.eval()

    def _get_tokens(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if hasattr(self.model, "get_intermediate_layers"):
            try:
                tokens = self.model.get_intermediate_layers(x, n=1, reshape=False)
                if isinstance(tokens, (list, tuple)):
                    tokens = tokens[0]
                if isinstance(tokens, torch.Tensor) and tokens.dim() == 3:
                    return tokens
            except Exception:
                return None
        if hasattr(self.model, "forward_features"):
            try:
                output = self.model.forward_features(x)
                if isinstance(output, dict):
                    tokens = output.get("x_norm_patchtokens")
                    if tokens is None:
                        tokens = output.get("patch_tokens")
                    if tokens is None:
                        tokens = output.get("x_norm")
                    if isinstance(tokens, torch.Tensor) and tokens.dim() == 3:
                        return tokens
            except Exception:
                return None
        return None

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        grid_h, grid_w = infer_grid_size(x, self.patch_size)
        num_patches = grid_h * grid_w

        tokens = self._get_tokens(x)
        if tokens is None:
            raise RuntimeError("Unable to extract tokens from DINOv3 backbone.")

        patch_tokens, cls_token, register_tokens = split_special_tokens(
            tokens,
            num_patches=num_patches,
            num_register_tokens=self.num_register_tokens,
        )

        global_embedding = patch_tokens.mean(dim=1)
        return BackboneOutput(
            global_embedding=global_embedding,
            patch_tokens=patch_tokens,
            grid_size=(grid_h, grid_w),
            cls_token=cls_token,
            register_tokens=register_tokens,
        )
