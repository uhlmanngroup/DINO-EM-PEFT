from pathlib import Path
import yaml
import os
import torch
import mlflow
import torch.nn as nn
from mlflow import log_metric, log_param, log_artifacts
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F

from dino_peft.datasets.paired_dirs_seg import PairedDirsSegDataset
from dino_peft.utils.transforms import em_seg_transforms
from dino_peft.models.backbone_dinov2 import DINOv2FeatureExtractor
from dino_peft.models.lora import inject_lora, lora_parameters
from dino_peft.models.head_seg1x1 import SegHead1x1


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
        # logits: [B,2,H,W], target: [B,H,W] (long, 0/1)
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
        lam = float(cfg.get("tversky_weight", 0.6))  # 0..1
        def _crit(logits, target):
            return (1.0 - lam) * ce(logits, target) + lam * tv(logits, target)
        return _crit
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


class SegTrainer:
    def __init__(self, cfg_path: str):
        # -------- config & device ----------
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.device = pick_device(self.cfg.get("device", "auto"))
        print(">> Using device:", self.device)

        self.out_dir = Path(self.cfg["out_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # -------- mlflow ----------
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:mlruns"))
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "default"))

        # Give the run a helpful name
        run_name = f"{self.cfg.get('dino_size','?')}_img{self.cfg['img_size'][0]}_{'lora' if self.cfg.get('use_lora',True) else 'head'}"
        self.mlflow_run = mlflow.start_run(run_name=run_name)

        # Log key params (avoid dumping giant dicts)
        log_param("dino_size", self.cfg.get("dino_size"))
        log_param("img_size",  str(self.cfg.get("img_size")))
        log_param("use_lora",  bool(self.cfg.get("use_lora", True)))
        log_param("lora_rank", int(self.cfg.get("lora_rank", 0)))
        log_param("lora_alpha",int(self.cfg.get("lora_alpha", 0)))
        log_param("batch_size",int(self.cfg.get("batch_size")))
        log_param("epochs",    int(self.cfg.get("epochs")))
        log_param("lr",        float(self.cfg.get("lr")))
        log_param("weight_decay", float(self.cfg.get("weight_decay")))
        log_param("loss", self.cfg.get("loss","ce"))
        log_param("class_weights", str(self.cfg.get("class_weights")))
        log_param("tversky", f"{self.cfg.get('tversky_alpha',0.7)},{self.cfg.get('tversky_beta',0.3)}")

        # -------- transforms ----------
        t = em_seg_transforms(tuple(self.cfg["img_size"]))

        # -------- datasets (single train/val pair) ----------
        if "train_img_dir" not in self.cfg or "train_mask_dir" not in self.cfg:
            raise ValueError("Config must provide train_img_dir and train_mask_dir.")

        self.train_ds = PairedDirsSegDataset(
            self.cfg["train_img_dir"],
            self.cfg["train_mask_dir"],
            img_size=self.cfg["img_size"],
            to_rgb=True,
            transform=t,
            binarize=bool(self.cfg.get("binarize", True)),
            binarize_threshold=int(self.cfg.get("binarize_threshold", 128)),
        )

        if "val_img_dir" in self.cfg and "val_mask_dir" in self.cfg:
            self.val_ds = PairedDirsSegDataset(
                self.cfg["val_img_dir"],
                self.cfg["val_mask_dir"],
                img_size=self.cfg["img_size"],
                to_rgb=True,
                transform=t,
                binarize=bool(self.cfg.get("binarize", True)),
                binarize_threshold=int(self.cfg.get("binarize_threshold", 128)),
            )
        else:
            # optional fallback: split a bit from train if no explicit val set provided
            val_ratio = float(self.cfg.get("val_ratio", 0.1))
            n = len(self.train_ds)
            n_val = max(1, int(round(n * val_ratio)))
            n_train = n - n_val
            g = torch.Generator().manual_seed(int(self.cfg.get("split_seed", 42)))
            self.train_ds, self.val_ds = random_split(self.train_ds, [n_train, n_val], generator=g)

        # -------- loaders ----------
        pin = (self.device.type == "cuda")
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=self.cfg["num_workers"],
            pin_memory=pin,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            num_workers=self.cfg["num_workers"],
            pin_memory=pin,
        )

        # -------- model ----------
        self.backbone = DINOv2FeatureExtractor(size=self.cfg["dino_size"], device=str(self.device))
        in_ch = self.backbone.embed_dim
        self.head = SegHead1x1(in_ch, self.cfg["num_classes"]).to(self.device)

        # -------- LoRA ----------
        self.lora_names = []
        if self.cfg.get("use_lora", True):
            self.lora_names = inject_lora(
                self.backbone.vit,
                target_substrings=self.cfg.get("lora_targets", ["attn.qkv", "attn.proj"]),
                r=int(self.cfg.get("lora_rank", 8)),
                alpha=int(self.cfg.get("lora_alpha", 16)),
            )
        # ensure any new LoRA modules are on our device
        self.backbone.to(self.device)

        # freeze base, enable LoRA + head
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in lora_parameters(self.backbone):
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

        # -------- optim / loss ----------
        params = list(self.head.parameters()) + list(lora_parameters(self.backbone))
        self.trainable_params = params  # <-- add this
        self.optimizer = torch.optim.AdamW(params, lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        self.criterion = build_criterion(self.cfg, device=self.device)
        self.epochs = int(self.cfg["epochs"])
        self.clip_grad_norm = float(self.cfg.get("clip_grad_norm", 0.0))

        # -------- AMP (device-aware) ----------
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
        with open(self.out_dir / "config_used.yaml", "w") as f:
            yaml.safe_dump(self.cfg, f)
        with open(self.out_dir / "lora_layers.txt", "w") as f:
            for n in self.lora_names:
                f.write(n + "\n")

    def _step(self, batch, train=True):
        imgs, masks = batch
        # Ensure target dtype is long indices for CE
        if masks.dtype != torch.long:
            masks = masks.long()

        imgs  = imgs.to(self.device, non_blocking=True)
        masks = masks.to(self.device, non_blocking=True)

        out_hw = masks.shape[-2:]
        self.optimizer.zero_grad(set_to_none=True)

        with self.autocast():
            feats  = self.backbone(imgs)               # (B, C, H', W')
            logits = self.head(feats, out_hw)          # (B, K, H, W)
            loss   = self.criterion(logits, masks)

        # Quick NaN/Inf guard before backward
        if not torch.isfinite(loss):
            print("Non-finite loss encountered; skipping step.")
            return float("nan"), logits

        if train:
            if self.scaler is not None and self.use_amp:
                # AMP path
                self.scaler.scale(loss).backward()

                # Unscale before clipping so clip sees true grads
                if self.clip_grad_norm and self.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=self.clip_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # FP32 path
                loss.backward()
                if self.clip_grad_norm and self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=self.clip_grad_norm)
                self.optimizer.step()

        return loss.item(), logits


    @torch.no_grad()
    def _colorize_mask(self, m: torch.Tensor, num_classes: int):
        """
        m: (B,H,W) long {0..K-1} -> (B,3,H,W) float in [0,1]
        simple palette: bg=black, fg=white for K=2; otherwise a few distinct colors
        """
        K = num_classes
        B, H, W = m.shape
        out = torch.zeros(B, 3, H, W, device=m.device, dtype=torch.float32)
        if K == 2:
            out[:, :, :, :] = 0.0
            out[:, 0] = (m == 1).float()  # white
            out[:, 1] = (m == 1).float()
            out[:, 2] = (m == 1).float()
        else:
            palette = torch.tensor([
                [0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
                [1,0,1], [0,1,1], [1,0.5,0], [0.5,0,1], [0.5,0.5,0.5]
            ], device=m.device, dtype=torch.float32)
            for k in range(min(K, palette.shape[0])):
                maskk = (m == k).unsqueeze(1).float()
                out += maskk * palette[k].view(1,3,1,1)
            out.clamp_(0,1)
        return out

    @torch.no_grad()
    def _save_preview(self, imgs, logits, masks, step_tag: str):
        grid_dir = self.out_dir / "previews"
        grid_dir.mkdir(exist_ok=True, parents=True)

        # move to CPU for saving (safe on CUDA/MPS)
        imgs_cpu  = imgs[:4].detach().cpu().clamp(0, 1)        # (B,3,h,w)
        preds_cpu = logits[:4].detach().argmax(1).cpu()        # (B,h,w)
        gts_cpu   = masks[:4].detach().cpu()                   # (B,h,w)

        # colorize on CPU
        pred_rgb = self._colorize_mask(preds_cpu, self.cfg["num_classes"])  # (B,3,H,W)
        gt_rgb   = self._colorize_mask(gts_cpu,   self.cfg["num_classes"])

        # size match
        H, W = gts_cpu.shape[-2:]
        if imgs_cpu.shape[-2:] != (H, W):
            imgs_cpu = F.interpolate(imgs_cpu, size=(H, W), mode="bilinear", align_corners=False)

        # save strips
        save_image(imgs_cpu,  grid_dir / f"{step_tag}_img.png",  nrow=4)
        save_image(pred_rgb,  grid_dir / f"{step_tag}_pred.png", nrow=4)
        save_image(gt_rgb,    grid_dir / f"{step_tag}_gt.png",   nrow=4)

        # triptych (img | pred | gt)
        trip = torch.cat([imgs_cpu, pred_rgb, gt_rgb], dim=0)
        grid = make_grid(trip, nrow=4)
        save_image(grid, grid_dir / f"{step_tag}_triptych.png")

    def train(self):
        best_val = float('inf')
        best_path = self.out_dir / "checkpoint_best.pt"

        for epoch in range(1, self.epochs + 1):
            # ---- TRAIN ----
            self.backbone.train(False)
            self.head.train(True)

            running = 0.0
            for batch in self.train_loader:
                loss, _ = self._step(batch, train=True)
                if isinstance(loss, float) and (loss != loss):  # NaN guard
                    continue
                running += loss
            avg_train = running / max(1, len(self.train_loader))

            # ---- VAL ----
            self.head.eval()
            val_loss = 0.0

            # accumulate pixel stats
            fg_gt_total = 0
            fg_pred_total = 0
            bg_gt_total = 0
            bg_pred_total = 0

            with torch.no_grad():
                for i, batch in enumerate(self.val_loader):
                    loss, logits = self._step(batch, train=False)
                    val_loss += loss

                    imgs, masks = batch
                    pred = logits.argmax(1)

                    fg_gt_total  += (masks == 1).sum().item()
                    fg_pred_total+= (pred  == 1).sum().item()
                    bg_gt_total  += (masks == 0).sum().item()
                    bg_pred_total+= (pred  == 0).sum().item()

                    if i == 0:
                        self._save_preview(imgs, logits, masks, f"ep{epoch:03d}")

            val_loss /= max(1, len(self.val_loader))
            print(f"[epoch {epoch}/{self.epochs}] train_loss={avg_train:.4f}  val_loss={val_loss:.4f}")
            print(f"          val FG: GT={fg_gt_total}  PRED={fg_pred_total} | BG: GT={bg_gt_total}  PRED={bg_pred_total}")

            # ---- MLflow scalars (AFTER computing val) ----
            log_metric("train/loss", float(avg_train), step=epoch)
            log_metric("val/loss",   float(val_loss),  step=epoch)
            log_metric("val/fg_gt_px",   int(fg_gt_total),   step=epoch)
            log_metric("val/fg_pred_px", int(fg_pred_total), step=epoch)
            log_metric("val/bg_gt_px",   int(bg_gt_total),   step=epoch)
            log_metric("val/bg_pred_px", int(bg_pred_total), step=epoch)

            # ---- save checkpoints: last + best ----
            ckpt = {
                "head": self.head.state_dict(),
                "backbone_lora": {k: v for k, v in self.backbone.state_dict().items() if "lora_" in k},
                "cfg": self.cfg,
                "epoch": epoch,
                "val_loss": float(val_loss),
            }
            torch.save(ckpt, self.out_dir / "checkpoint_last.pt")
            if val_loss < best_val:
                best_val = float(val_loss)
                torch.save(ckpt, best_path)
                print(f"[info] New best -> {best_path} (val_loss={best_val:.4f})")
                # optional: track best in MLflow
                log_metric("val/best_loss", best_val, step=epoch)

            # light artifact logging (previews) every 5 epochs is fine
            try:
                if epoch % 5 == 0:
                    log_artifacts(str(self.out_dir / "previews"), artifact_path="previews")
            except Exception as e:
                print("[mlflow] preview artifact upload skipped:", e)

        # ---- end of training: log final artifacts once ----
        try:
            # if you also write history.csv/loss_curve.png, log them here
            log_artifacts(str(self.out_dir), artifact_path="run_artifacts")
        except Exception as e:
            print("[mlflow] final artifact upload skipped:", e)

        try:
            mlflow.end_run()
        except Exception as e:
            print("[mlflow] end_run() failed:", e)

