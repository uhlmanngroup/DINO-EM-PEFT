from __future__ import annotations

from pathlib import Path
import inspect
from math import sqrt
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import (
    BackboneAdapter,
    BackboneOutput,
    infer_grid_size,
    infer_num_register_tokens,
    infer_patch_size,
    split_special_tokens,
)

try:  # pragma: no cover - optional dependency
    import open_clip
except Exception as exc:  # pragma: no cover
    open_clip = None
    _OPENCLIP_IMPORT_ERR = exc
else:  # pragma: no cover
    _OPENCLIP_IMPORT_ERR = None


def _require_openclip() -> None:
    if open_clip is None:
        raise ImportError(
            "open_clip is not available. Install open-clip-torch to use the OpenCLIP backbone."
        ) from _OPENCLIP_IMPORT_ERR


def _resolve_pretrained_and_weights(
    pretrained: str | None,
    weights: str | None,
) -> tuple[str | None, str | None]:
    weights_path = None
    pretrained_tag = pretrained or None
    if pretrained:
        candidate = Path(pretrained).expanduser()
        if candidate.exists():
            weights_path = str(candidate)
            pretrained_tag = None
    if weights:
        candidate = Path(weights).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"OpenCLIP weights not found: {candidate}")
        weights_path = str(candidate)
    return pretrained_tag, weights_path


def _load_openclip_model(
    model_name: str,
    pretrained: str | None,
    weights: str | None,
    device: torch.device,
) -> nn.Module:
    _require_openclip()
    pretrained_tag, weights_path = _resolve_pretrained_and_weights(pretrained, weights)
    model = open_clip.create_model(model_name, pretrained=pretrained_tag)
    if weights_path:
        if hasattr(open_clip, "load_checkpoint"):
            open_clip.load_checkpoint(model, weights_path)
        else:
            state = torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict):
                state = state.get("state_dict", state.get("model", state))
            model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def _infer_embed_dim(model: nn.Module) -> int:
    for attr in ("embed_dim", "width", "hidden_dim", "dim"):
        val = getattr(model, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    proj = getattr(model, "proj", None)
    if hasattr(proj, "shape") and len(proj.shape) >= 2:
        return int(proj.shape[0])
    raise ValueError("Unable to infer embed_dim from OpenCLIP visual model.")


def _extract_tokens(output: object) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor) and output.dim() == 3:
        return output
    if isinstance(output, dict):
        for key in (
            "x",
            "x_norm",
            "x_norm_patchtokens",
            "patch_tokens",
            "tokens",
            "last_hidden_state",
        ):
            tokens = output.get(key)
            if isinstance(tokens, torch.Tensor) and tokens.dim() == 3:
                return tokens
    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, torch.Tensor) and item.dim() == 3:
                return item
    return None


def _shape_tokens(tokens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tokens is None:
        return None
    if tokens.dim() == 2:
        return None
    if tokens.dim() != 3:
        return None
    return tokens


class OpenCLIPAdapter(BackboneAdapter):
    def __init__(
        self,
        model_name: str,
        device: str | torch.device,
        *,
        pretrained: str | None = None,
        weights: str | None = None,
    ) -> None:
        super().__init__(name="openclip", variant=model_name)
        device_obj = device if isinstance(device, torch.device) else torch.device(device)
        clip_model = _load_openclip_model(model_name, pretrained, weights, device_obj)
        visual = getattr(clip_model, "visual", None)
        if visual is None:
            raise RuntimeError("OpenCLIP model does not expose a visual tower.")

        self.model = visual
        self.embed_dim = _infer_embed_dim(visual)
        self.patch_size = infer_patch_size(visual, fallback=14)
        self.num_register_tokens = infer_num_register_tokens(visual)
        self._pos_embed_info = None
        self._pos_embed_last_grid = None
        self._pos_embed_base = None
        self._pos_embed_base_grid = None
        self._pos_embed_cache = {}
        self.to(device_obj)
        self.eval()

    def _maybe_resize_pos_embed(self, x: torch.Tensor) -> None:
        grid_h, grid_w = infer_grid_size(x, self.patch_size)
        new_grid = (grid_h, grid_w)
        if self._pos_embed_last_grid == new_grid:
            return

        info = self._pos_embed_info or self._resolve_pos_embed()
        pos_embed = info["tensor"]
        has_cls = info["has_cls"]
        pos_embed_dim = pos_embed.dim()

        if self._pos_embed_base is None:
            base_tokens = self._normalize_pos_embed(pos_embed)
            self._pos_embed_base = base_tokens.detach().clone()
            self._pos_embed_base_grid = info["grid"]
            cached = base_tokens if pos_embed_dim == 3 else base_tokens.squeeze(0)
            self._pos_embed_cache[self._pos_embed_base_grid] = cached

        cached = self._pos_embed_cache.get(new_grid)
        if cached is None:
            base_tokens = self._pos_embed_base
            if base_tokens.device != pos_embed.device or base_tokens.dtype != pos_embed.dtype:
                base_tokens = base_tokens.to(device=pos_embed.device, dtype=pos_embed.dtype)
                self._pos_embed_base = base_tokens
            base_grid = self._pos_embed_base_grid or info["grid"]
            patch_tokens = base_tokens[:, 1:, :] if has_cls else base_tokens
            patch_tokens = patch_tokens.reshape(
                1, base_grid[0], base_grid[1], -1
            ).permute(0, 3, 1, 2)
            resized = F.interpolate(patch_tokens, size=new_grid, mode="bicubic", align_corners=False)
            resized = resized.permute(0, 2, 3, 1).reshape(1, new_grid[0] * new_grid[1], -1)
            if has_cls:
                tokens = torch.cat([base_tokens[:, :1, :], resized], dim=1)
            else:
                tokens = resized
            cached = tokens if pos_embed_dim == 3 else tokens.squeeze(0)
            self._pos_embed_cache[new_grid] = cached
            print(
                "[openclip] resized pos_embed "
                f"{base_grid[0]}x{base_grid[1]} -> {new_grid[0]}x{new_grid[1]} cls={has_cls}"
            )

        if cached.device != pos_embed.device or cached.dtype != pos_embed.dtype:
            cached = cached.to(device=pos_embed.device, dtype=pos_embed.dtype)
            self._pos_embed_cache[new_grid] = cached

        self._assign_pos_embed(info, cached)
        info["grid"] = new_grid
        self._pos_embed_last_grid = new_grid

    def _normalize_pos_embed(self, pos_embed: torch.Tensor) -> torch.Tensor:
        if pos_embed.dim() == 2:
            return pos_embed.unsqueeze(0)
        if pos_embed.dim() == 3 and pos_embed.shape[0] == 1:
            return pos_embed
        raise RuntimeError(
            f"Unsupported pos_embed shape {tuple(pos_embed.shape)} for OpenCLIP visual model."
        )

    def _resolve_pos_embed(self) -> dict:
        candidates = []
        for name, param in self.model.named_parameters(recurse=True):
            if param is None:
                continue
            candidates.append((name, param, True))
        for name, buf in self.model.named_buffers(recurse=True):
            if buf is None:
                continue
            candidates.append((name, buf, False))

        scored = []
        for name, tensor, is_param in candidates:
            if tensor.dim() not in (2, 3):
                continue
            if tensor.shape[-1] != self.embed_dim:
                continue
            tokens = tensor.shape[1] if tensor.dim() == 3 else tensor.shape[0]
            has_cls, grid = self._infer_pos_embed_grid(tokens)
            if grid is None:
                continue
            score = 0
            lower = name.lower()
            if "pos" in lower:
                score += 2
            if "embed" in lower:
                score += 1
            if "positional" in lower:
                score += 2
            scored.append((score, name, tensor, is_param, has_cls, grid))

        if not scored:
            sample = [f"{name}: {tuple(t.shape)}" for name, t, _ in candidates][:20]
            hint = "\n  ".join(sample) if sample else "(no candidate tensors found)"
            raise RuntimeError(
                "OpenCLIP pos_embed not found or unsupported. "
                f"model={self.variant}\n"
                f"Candidate tensors:\n  {hint}\n"
                "Consider checking OpenCLIP version or adding a custom resolver."
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        _, name, tensor, is_param, has_cls, grid = scored[0]
        info = {
            "name": name,
            "tensor": tensor,
            "is_param": is_param,
            "has_cls": has_cls,
            "grid": grid,
        }
        self._pos_embed_info = info
        return info

    def _infer_pos_embed_grid(self, tokens: int) -> tuple[bool, tuple[int, int] | None]:
        image_size = getattr(self.model, "image_size", None)
        if image_size is not None:
            if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
                grid_h = int(image_size[0]) // self.patch_size
                grid_w = int(image_size[1]) // self.patch_size
            elif isinstance(image_size, int):
                grid_h = grid_w = int(image_size) // self.patch_size
            else:
                grid_h = grid_w = None
            if grid_h and grid_w:
                if tokens == grid_h * grid_w:
                    return False, (grid_h, grid_w)
                if tokens == grid_h * grid_w + 1:
                    return True, (grid_h, grid_w)

        patch_tokens = tokens - 1
        root = int(sqrt(patch_tokens))
        if root * root == patch_tokens:
            return True, (root, root)
        root = int(sqrt(tokens))
        if root * root == tokens:
            return False, (root, root)
        return False, None

    def _assign_pos_embed(self, info: dict, new_tensor: torch.Tensor) -> None:
        name = info["name"]
        if info["is_param"]:
            param = dict(self.model.named_parameters(recurse=True)).get(name)
            if param is None:
                raise RuntimeError(f"OpenCLIP pos_embed parameter '{name}' not found.")
            param.data = new_tensor
            info["tensor"] = param
            return
        buffer = dict(self.model.named_buffers(recurse=True)).get(name)
        if buffer is None:
            raise RuntimeError(f"OpenCLIP pos_embed buffer '{name}' not found.")
        module = self.model
        parts = name.split(".")
        for part in parts[:-1]:
            module = getattr(module, part)
        module._buffers[parts[-1]] = new_tensor
        info["tensor"] = new_tensor

    def _get_tokens(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if hasattr(self.model, "forward_features"):
            try:
                fn = self.model.forward_features
                sig = inspect.signature(fn)
                kwargs = {}
                for key in (
                    "return_all_tokens",
                    "return_tokens",
                    "return_patch_tokens",
                    "output_tokens",
                    "return_all_features",
                ):
                    if key in sig.parameters:
                        kwargs[key] = True
                output = fn(x, **kwargs) if kwargs else fn(x)
                tokens = _shape_tokens(_extract_tokens(output))
                if tokens is not None:
                    return tokens
            except Exception:
                pass

        if hasattr(self.model, "forward_intermediates"):
            try:
                fn = self.model.forward_intermediates
                sig = inspect.signature(fn)
                kwargs = {}
                if "indices" in sig.parameters:
                    kwargs["indices"] = 1
                if "output_fmt" in sig.parameters:
                    kwargs["output_fmt"] = "NLC"
                if "output_extra_tokens" in sig.parameters:
                    kwargs["output_extra_tokens"] = True
                output = fn(x, **kwargs) if kwargs else fn(x)
                if isinstance(output, dict):
                    intermediates = output.get("image_intermediates")
                    if isinstance(intermediates, list) and intermediates:
                        patches = intermediates[-1]
                        if patches.dim() == 3:
                            prefix = output.get("image_intermediates_prefix")
                            if isinstance(prefix, list) and prefix:
                                cls = prefix[-1]
                                if cls.dim() == 3:
                                    return torch.cat([cls, patches], dim=1)
                            return patches
                tokens = _shape_tokens(_extract_tokens(output))
                if tokens is not None:
                    return tokens
            except Exception:
                pass

        if hasattr(self.model, "get_intermediate_layers"):
            try:
                output = self.model.get_intermediate_layers(x, n=1, reshape=False)
                tokens = _shape_tokens(_extract_tokens(output))
                if tokens is not None:
                    return tokens
            except Exception:
                pass

        for flag in ("return_all_tokens", "return_tokens", "return_all_features", "output_tokens"):
            try:
                output = self.model(x, **{flag: True})
            except TypeError:
                continue
            tokens = _shape_tokens(_extract_tokens(output))
            if tokens is not None:
                return tokens

        tokens = self._manual_vit_tokens(x)
        if tokens is not None:
            return tokens
        return None

    def _manual_vit_tokens(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        required = ("conv1", "class_embedding", "positional_embedding", "ln_pre", "transformer")
        if not all(hasattr(self.model, attr) for attr in required):
            return None
        model = self.model
        try:
            x = model.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls = model.class_embedding
            if cls.dim() == 1:
                cls = cls.unsqueeze(0)
            cls = cls.to(dtype=x.dtype, device=x.device)
            cls = cls.expand(x.shape[0], 1, -1)
            x = torch.cat([cls, x], dim=1)
            pos = model.positional_embedding
            if pos is not None:
                if pos.dim() == 2:
                    pos = pos.unsqueeze(0)
                if pos.shape[1] == x.shape[1]:
                    x = x + pos.to(dtype=x.dtype, device=x.device)
            x = model.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = model.transformer(x)
            x = x.permute(1, 0, 2)
            if hasattr(model, "ln_post"):
                x = model.ln_post(x)
            return _shape_tokens(x)
        except Exception:
            return None

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        self._maybe_resize_pos_embed(x)
        grid_h, grid_w = infer_grid_size(x, self.patch_size)
        num_patches = grid_h * grid_w

        tokens = self._get_tokens(x)
        if tokens is None:
            raise RuntimeError("Unable to extract tokens from OpenCLIP visual backbone.")

        try:
            patch_tokens, cls_token, register_tokens = split_special_tokens(
                tokens,
                num_patches=num_patches,
                num_register_tokens=self.num_register_tokens,
            )
        except ValueError as exc:
            raise RuntimeError(
                "OpenCLIP token count mismatch. This usually means positional embedding "
                "interpolation is unsupported for the requested image size."
            ) from exc

        if cls_token is not None and cls_token.numel() > 0:
            global_embedding = cls_token.squeeze(1)
        else:
            global_embedding = patch_tokens.mean(dim=1)
        return BackboneOutput(
            global_embedding=global_embedding,
            patch_tokens=patch_tokens,
            grid_size=(grid_h, grid_w),
            cls_token=cls_token,
            register_tokens=register_tokens,
        )
