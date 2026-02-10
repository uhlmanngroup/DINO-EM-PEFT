from __future__ import annotations

from typing import Optional, Tuple

import torch

from dino_peft.models.backbone_dinov2 import DINOv2FeatureExtractor

from .base import BackboneAdapter, BackboneOutput, infer_num_register_tokens, split_special_tokens


class DINOv2Adapter(BackboneAdapter):
    def __init__(self, variant: str, device: str | torch.device) -> None:
        super().__init__(name="dinov2", variant=variant)
        self.feature_extractor = DINOv2FeatureExtractor(size=variant, device=device)
        self.model = self.feature_extractor.vit
        self.embed_dim = self.feature_extractor.embed_dim
        self.patch_size = self.feature_extractor.patch_size
        self.num_register_tokens = infer_num_register_tokens(self.model)
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
        return None

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        feats = self.feature_extractor.forward_features(x)
        grid_h, grid_w = feats.shape[-2], feats.shape[-1]
        num_patches = grid_h * grid_w

        cls_token = None
        register_tokens = None
        tokens = self._get_tokens(x)
        if tokens is not None:
            patch_tokens, cls_token, register_tokens = split_special_tokens(
                tokens,
                num_patches=num_patches,
                num_register_tokens=self.num_register_tokens,
            )
        else:
            patch_tokens = feats.flatten(2).transpose(1, 2)

        global_embedding = patch_tokens.mean(dim=1)
        return BackboneOutput(
            global_embedding=global_embedding,
            patch_tokens=patch_tokens,
            grid_size=(grid_h, grid_w),
            cls_token=cls_token,
            register_tokens=register_tokens,
        )
