import yaml
import torch
import torch.nn as nn
import monai
from torch.utils.data import DataLoader
import torch.nn.functional as F
from copy import deepcopy

from dino_peft.datasets.lucchi_seg import LucchiSegDataset
from dino_peft.datasets.droso_seg import DrosoSegDataset
from dino_peft.datasets.paired_dirs_seg import PairedDirsSegDataset
from dino_peft.utils.transforms import em_seg_transforms, denorm_imagenet
from dino_peft.utils.viz import colorize_mask
from dino_peft.utils.plots import save_triptych_grid
from dino_peft.backbones import (
    build_backbone,
    patch_tokens_to_grid,
    resolve_backbone_cfg,
    resolve_preprocess_cfg,
)
from dino_peft.models.lora import apply_peft, lora_parameters
from dino_peft.models.head_seg1x1 import SegHeadDeconv
from dino_peft.utils.paths import setup_run_dir, write_run_info, update_metrics
from dino_peft.utils.image_size import DEFAULT_IMG_SIZE_CFG


def pick_device(cfg_device: str | None):
    if cfg_device and cfg_device.lower() != "auto":
        return torch.device(cfg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _make_ce(class_weights, device):
    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32, device=device)
    else:
        w = None
    return nn.CrossEntropyLoss(weight=w)

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        p_fg = probs[:, 1, ...]
        t_fg = (target == 1).float()
        tp = (p_fg * t_fg).sum(dim=(1,2))
        fp = (p_fg * (1.0 - t_fg)).sum(dim=(1,2))
        fn = ((1.0 - p_fg) * t_fg).sum(dim=(1,2))
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1.0 - tversky.mean()

def build_criterion(cfg, device):
    loss_name = cfg.get("loss", "ce")
    if loss_name == "ce":
        return _make_ce(cfg.get("class_weights"), device)
    elif loss_name == "ce_tversky":
        ce = _make_ce(cfg.get("class_weights"), device)
        tv = TverskyLoss(
            alpha=cfg.get("tversky_alpha", 0.7),
            beta=cfg.get("tversky_beta", 0.3),
            eps=cfg.get("tversky_eps", 1e-6),
        )
        lam = float(cfg.get("tversky_weight", 0.6))
        def _crit(logits, target):
            return (1.0 - lam) * ce(logits, target) + lam * tv(logits, target)
        return _crit
    elif loss_name == "dice":
        return monai.losses.DiceLoss(softmax=True, to_onehot_y=True, include_background=True)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

class SegTrainer:
    def __init__(self, cfg_path: str):
        # -------- config ----------
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.img_size_cfg = deepcopy(self.cfg.get("img_size", DEFAULT_IMG_SIZE_CFG))
        if "img_size" not in self.cfg:
            print("[SegTrainer] img_size not set in config; defaulting to longest_edge=1022.")
        self.cfg["img_size"] = deepcopy(self.img_size_cfg)

        # --- seed (repro) ---
        seed = int(self.cfg.get("seed", 0))
        import random, numpy as np
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.seed = seed

        # -------- device & out ----------
        self.device = pick_device(self.cfg.get("device", "auto"))
        print(">> Using device:", self.device)

        task_type = self.cfg.get("task_type", "seg")
        self.out_dir = setup_run_dir(
            self.cfg,
            task_type=task_type,
            subdirs=("logs", "ckpts", "figs", "figs/previews"),
        )
        self.ckpt_dir = self.out_dir / "ckpts"
        self.fig_dir = self.out_dir / "figs"
        self.previews_dir = self.fig_dir / "previews"
        backbone_cfg = resolve_backbone_cfg(self.cfg)
        self.backbone_cfg = backbone_cfg
        preprocess_cfg = resolve_preprocess_cfg(self.cfg, default_img_size=self.img_size_cfg)
        self.preprocess_cfg = preprocess_cfg
        write_run_info(
            self.out_dir,
            {
                "task_type": task_type,
                "device": str(self.device),
                "img_size": self.cfg.get("img_size"),
                "backbone_name": backbone_cfg.get("name"),
                "backbone_variant": backbone_cfg.get("variant"),
                "preprocess_preset": preprocess_cfg.get("preset"),
            },
        )

        # -------- dataset selection ----------
        dataset_cfg = self.cfg.get("dataset", {})
        dataset_type = str(dataset_cfg.get("type", "lucchi")).lower()
        dataset_params = dict(dataset_cfg.get("params", {}))
        dataset_map = {
            "lucchi": LucchiSegDataset,
            "droso": DrosoSegDataset,
            "paired": PairedDirsSegDataset,
        }
        DatasetClass = dataset_map.get(dataset_type)
        if DatasetClass is None:
            raise ValueError(
                f"Unsupported dataset.type '{dataset_type}'. "
                "Use 'lucchi', 'droso', or 'paired'."
            )
        if dataset_type == "lucchi":
            dataset_params.setdefault("recursive", False)
            dataset_params.setdefault("zfill_width", 4)
            dataset_params.setdefault("image_prefix", "mask")
        elif dataset_type == "droso":
            dataset_params.setdefault("recursive", True)

        def _build_dataset(img_dir, mask_dir, transform):
            kwargs = {
                "img_size": self.img_size_cfg,
                "to_rgb": True,
                "transform": transform,
                "binarize": bool(self.cfg.get("binarize", True)),
                "binarize_threshold": int(self.cfg.get("binarize_threshold", 128)),
            }
            kwargs.update(dataset_params)
            return DatasetClass(
                img_dir,
                mask_dir,
                **kwargs,
            )

        t_train = em_seg_transforms()   # deterministic pipeline (resize handled in dataset)
        t_val   = em_seg_transforms()

        # -------- base dataset (NO transform) ----------
        base_ds = _build_dataset(
            self.cfg["train_img_dir"],
            self.cfg["train_mask_dir"],
            transform=None,
        )

        # -------- 10% validation split ----------
        val_ratio = float(self.cfg.get("val_ratio", 0.1))
        n = len(base_ds)
        n_val = max(1, int(round(n * val_ratio)))
        n_train = n - n_val
        g = torch.Generator().manual_seed(int(self.cfg.get("split_seed", 42)))
        perm = torch.randperm(n, generator=g).tolist()
        train_idx, val_idx = perm[:n_train], perm[n_train:]

        def make_subset_dataset(src_ds, index_list, transform):
            ds = _build_dataset(
                self.cfg["train_img_dir"],
                self.cfg["train_mask_dir"],
                transform=transform,
            )
            ds.pairs = [src_ds.pairs[i] for i in index_list]
            return ds

        # -------- final datasets ----------
        self.train_ds = make_subset_dataset(base_ds, train_idx, t_train)
        self.val_ds   = make_subset_dataset(base_ds, val_idx,   t_val)   # no transform -> no aug

        # -------- loaders ----------
        pin = (self.device.type == "cuda")
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=self.cfg["num_workers"],
            pin_memory=pin,
            collate_fn=self._pad_collate,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            num_workers=self.cfg["num_workers"],
            pin_memory=pin,
            collate_fn=self._pad_collate,
        )

        # -------- model ----------
        self.backbone = build_backbone(self.backbone_cfg, device=self.device)
        in_ch = self.backbone.embed_dim
        self.head = SegHeadDeconv(
            in_ch=in_ch,
            num_classes=self.cfg["num_classes"],
            n_ups=4,
            base_ch=512,
        ).to(self.device)

        # -------- LoRA ----------
        self.lora_names = []
        audit = apply_peft(
            self.backbone.model,
            self.cfg,
            run_dir=self.out_dir,
            backbone_info=self.backbone_cfg,
        )
        self.lora_enabled = audit is not None
        if audit is not None:
            self.lora_names = audit.targets
        self.backbone.to(self.device)  # ensure LoRA modules on device
        for p in self.head.parameters():
            p.requires_grad = True

        # -------- optim / loss ----------
        params = list(self.head.parameters()) + list(lora_parameters(self.backbone.model))
        self.trainable_params = params
        self.optimizer = torch.optim.AdamW(params, lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        print(f"[params] trainable={sum(p.numel() for p in self.trainable_params):,}")
        print(
            "[warn] unexpected trainable in backbone:",
            [n for n, p in self.backbone.model.named_parameters() if p.requires_grad and "lora_" not in n][:15],
        )
        self.criterion = build_criterion(self.cfg, device=self.device)
        self.epochs = int(self.cfg["epochs"])
        self.patience = max(1, int(self.cfg.get("patience", 20)))
        self.clip_grad_norm = float(self.cfg.get("clip_grad_norm", 0.0))

        # -------- AMP ----------
        self.use_amp = bool(self.cfg.get("amp", True))
        if self.device.type == "cuda":
            from torch.amp import GradScaler, autocast
            self.scaler = GradScaler("cuda", enabled=self.use_amp)
            self.autocast = lambda: autocast("cuda", enabled=self.use_amp)
        elif self.device.type == "mps":
            from torch import autocast
            self.scaler = None
            self.autocast = lambda: autocast(device_type="mps", enabled=self.use_amp)
        else:
            from torch import autocast
            self.scaler = None
            self.autocast = lambda: autocast(device_type="cpu", enabled=False)

        # -------- save config & lora list ----------
        with open(self.out_dir / "lora_layers.txt", "w") as f:
            for n in self.lora_names:
                f.write(n + "\n")

    @staticmethod
    def _pad_collate(batch):
        imgs, masks, names = zip(*batch)
        max_h = max(img.shape[-2] for img in imgs)
        max_w = max(img.shape[-1] for img in imgs)
        padded_imgs = []
        padded_masks = []
        for img, mask in zip(imgs, masks):
            c = img.shape[0]
            h, w = img.shape[-2], img.shape[-1]
            pad_img = img.new_zeros((c, max_h, max_w))
            pad_img[:, :h, :w] = img
            padded_imgs.append(pad_img)

            mh, mw = mask.shape[-2], mask.shape[-1]
            pad_mask = mask.new_full((max_h, max_w), 0)
            pad_mask[:mh, :mw] = mask
            padded_masks.append(pad_mask)
        imgs_tensor = torch.stack(padded_imgs)
        masks_tensor = torch.stack(padded_masks)
        return imgs_tensor, masks_tensor, list(names)

    def train(self):
        best_val = float('inf')
        best_epoch = 0
        val_loss = float('inf')
        avg_train = float('inf')
        best_path = self.ckpt_dir / "best_model.pt"
        last_path = self.ckpt_dir / "last_model.pt"

        img_tag = self.cfg.get("img_size")
        if isinstance(img_tag, (list, tuple)) and img_tag:
            img_tag_str = f"{img_tag[0]}x{img_tag[1]}" if len(img_tag) > 1 else str(img_tag[0])
        elif isinstance(img_tag, dict):
            tgt = img_tag.get("target") or img_tag.get("target_long_edge")
            mode = img_tag.get("mode", "var")
            img_tag_str = f"{mode}-{tgt}" if tgt else mode
        else:
            img_tag_str = str(img_tag)
        backbone_tag = f"{self.backbone_cfg.get('name')}-{self.backbone_cfg.get('variant')}"
        run_name = (
            f"{backbone_tag}_img{img_tag_str}_"
            f"{'lora' if getattr(self, 'lora_enabled', False) else 'head'}"
        )
        print(f"[train] run_name={run_name}")

        # --------- training loop ----------
        epochs_since_improve = 0
        epochs_completed = 0
        best_train_loss = float("inf")
        last_train_loss = float("inf")
        last_val_loss = float("inf")
        for epoch in range(1, self.epochs + 1):
            self.backbone.train(False)
            self.head.train(True)

            running = 0.0
            # TRAIN
            for imgs, masks, _ in self.train_loader:
                masks = masks.long()
                imgs = imgs.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                if self.lora_enabled:
                    out = self.backbone(imgs)
                else:
                    with torch.no_grad():
                        out = self.backbone(imgs)
                feats = patch_tokens_to_grid(out)
                logits = self.head(feats, out_hw=masks.shape[-2:])

                loss = self.criterion(logits, masks.unsqueeze(1)) # From (B, H, W) to (B, K, H, W)  as expected by Monai DiceLoss
                
                # backward
                loss.backward()
                if self.clip_grad_norm and self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=self.clip_grad_norm)
                self.optimizer.step()
                running += loss.item()
            avg_train = running / max(1, len(self.train_loader))

            # VAL
            self.head.eval()
            val_loss = 0.0
            fg_gt_total = fg_pred_total = 0
            bg_gt_total = bg_pred_total = 0

            with torch.no_grad():
                for i, (imags, masks, names) in enumerate(self.val_loader):
                    masks = masks.long()
                    imgs = imags.to(self.device, non_blocking=True)
                    masks = masks.to(self.device, non_blocking=True)

                    out = self.backbone(imgs)
                    feats = patch_tokens_to_grid(out)
                    logits = self.head(feats, out_hw=masks.shape[-2:])
                    loss = self.criterion(logits, masks.unsqueeze(1)) # From (B, H, W) to (B, K, H, W)  as expected by Monai DiceLoss
                    val_loss += float(loss)

                    pred = logits.argmax(1)
                    fg_gt_total  += (masks == 1).sum().item()
                    fg_pred_total+= (pred  == 1).sum().item()
                    bg_gt_total  += (masks == 0).sum().item()
                    bg_pred_total+= (pred  == 0).sum().item()

                    if i == 0:
                        # ---- Save a labeled triptych (first batch only) ----
                        grid_dir = self.previews_dir
                        grid_dir.mkdir(parents=True, exist_ok=True)

                        # Show up to K samples
                        K = min(4, imgs.size(0))

                        # (B, H, W) int class indices
                        preds_cpu = logits[:K].detach().argmax(1).cpu()
                        gts_cpu   = masks[:K].detach().cpu()

                        # de-normalize images for visualization
                        imgs_vis = denorm_imagenet(imags[:K].detach().cpu()).clamp(0, 1)  # use original imags from loader

                        # resize imgs to match mask size if needed
                        H, W = gts_cpu.shape[-2:]
                        if imgs_vis.shape[-2:] != (H, W):
                            imgs_vis = F.interpolate(imgs_vis, size=(H, W), mode="bilinear", align_corners=False)

                        # colorize masks (RGB tensors, [0,1])
                        pred_rgb = colorize_mask(preds_cpu, self.cfg["num_classes"])
                        gt_rgb   = colorize_mask(gts_cpu,   self.cfg["num_classes"])

                        # column titles from dataset names
                        col_names = [str(n) for n in list(names)[:K]]

                        # pack samples
                        samples = []
                        for j in range(K):
                            samples.append({
                                "image": imgs_vis[j],  # CxHxW tensor is fine
                                "gt":    gt_rgb[j],    # already RGB
                                "pred":  pred_rgb[j],  # already RGB
                                "name":  col_names[j],
                            })

                        save_triptych_grid(
                            samples,
                            out_path=str(grid_dir / f"ep{epoch:03d}_triptych.png"),
                            title=f"Epoch {epoch} — Validation Triptychs"
                        )

            val_loss /= max(1, len(self.val_loader))
            last_train_loss = float(avg_train)
            last_val_loss = float(val_loss)
            epochs_completed = epoch
            print(f"[epoch {epoch}/{self.epochs}] train_loss={avg_train:.4f}  val_loss={val_loss:.4f}")

            # Checkpoints
            ckpt = {
                "head": self.head.state_dict(),
                "backbone_lora": {k: v for k, v in self.backbone.model.state_dict().items() if "lora_" in k},
                "cfg": self.cfg,
                "epoch": int(epoch),
                "val_loss": float(val_loss),
            }

            torch.save(ckpt, last_path)
            print(f"[ckpt] wrote {last_path.name}")

            if val_loss < best_val:
                best_val = float(val_loss)
                best_epoch = epoch
                best_train_loss = float(avg_train)
                epochs_since_improve = 0
                torch.save(ckpt, best_path)
                print(f"[ckpt] NEW BEST -> {best_path.name} (val_loss={best_val:.4f})")
            else:
                epochs_since_improve += 1

            if epochs_since_improve >= self.patience:
                print(f"[early_stop] no val improvement for {self.patience} epochs — stopping at epoch {epoch}")
                break
        update_metrics(
            self.out_dir,
            "train",
            {
                "best_val_loss": float(best_val),
                "best_epoch": int(best_epoch),
                "best_train_loss": float(best_train_loss),
                "last_val_loss": float(last_val_loss),
                "last_train_loss": float(last_train_loss),
                "last_epoch": int(epochs_completed),
                "max_epochs": int(self.epochs),
                "patience": int(self.patience),
                "seed": int(self.seed),
            },
        )
