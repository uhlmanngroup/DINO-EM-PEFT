from pathlib import Path
import yaml
import os
import torch
import mlflow
import torch.nn as nn
import monai
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from mlflow.tracking import MlflowClient  # <-- needed

from dino_peft.datasets.paired_dirs_seg import PairedDirsSegDataset
from dino_peft.utils.transforms import em_seg_transforms, denorm_imagenet
from dino_peft.models.backbone_dinov2 import DINOv2FeatureExtractor
from dino_peft.models.lora import inject_lora, lora_parameters
from dino_peft.models.head_seg1x1 import SegHeadDeconv


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
        return monai.losses.DiceLoss(to_onehot_y=False, sigmoid=True)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

class SegTrainer:
    def __init__(self, cfg_path: str):
        # -------- config ----------
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

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

        self.out_dir = Path(self.cfg["out_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # -------- transforms ----------
        t_train = em_seg_transforms(tuple(self.cfg["img_size"]))   # your current (deterministic) pipeline
        t_val   = None                                             # simplest: no transform in val

        # -------- base dataset (NO transform) ----------
        base_ds = PairedDirsSegDataset(
            self.cfg["train_img_dir"], 
            self.cfg["train_mask_dir"],
            img_size=self.cfg["img_size"], 
            to_rgb=True, 
            transform=None,
            binarize=bool(self.cfg.get("binarize", True)),
            binarize_threshold=int(self.cfg.get("binarize_threshold", 128)),
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
            ds = PairedDirsSegDataset(
                self.cfg["train_img_dir"], self.cfg["train_mask_dir"],
                img_size=self.cfg["img_size"], to_rgb=True, transform=transform,
                binarize=bool(self.cfg.get("binarize", True)),
                binarize_threshold=int(self.cfg.get("binarize_threshold", 128)),
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
        self.head = SegHeadDeconv(in_ch=in_ch, num_classes=self.cfg["num_classes"], n_ups=4, base_ch=512).to(self.device)

        # -------- LoRA ----------
        self.lora_names = []
        if self.cfg.get("use_lora", True):
            self.lora_names = inject_lora(
                self.backbone.vit,
                target_substrings=self.cfg.get("lora_targets", ["attn.qkv", "attn.proj"]),
                r=int(self.cfg.get("lora_rank", 8)),
                alpha=int(self.cfg.get("lora_alpha", 16)),
            )
        self.backbone.to(self.device)  # ensure LoRA modules on device

        # freeze base, enable LoRA + head
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in lora_parameters(self.backbone):
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

        # -------- optim / loss ----------
        params = list(self.head.parameters()) + list(lora_parameters(self.backbone))
        self.trainable_params = params
        self.optimizer = torch.optim.AdamW(params, lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        self.criterion = build_criterion(self.cfg, device=self.device)
        self.epochs = int(self.cfg["epochs"])
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
        with open(self.out_dir / "config_used.yaml", "w") as f:
            yaml.safe_dump(self.cfg, f)
        with open(self.out_dir / "lora_layers.txt", "w") as f:
            for n in self.lora_names:
                f.write(n + "\n")

    def _step(self, batch, train=True):
        imgs, masks = batch
        if masks.dtype != torch.long:
            masks = masks.long()

        imgs  = imgs.to(self.device, non_blocking=True)
        masks = masks.to(self.device, non_blocking=True)

        out_hw = masks.shape[-2:]
        self.optimizer.zero_grad(set_to_none=True)

        with self.autocast():
            feats  = self.backbone(imgs)               # (B, C, H', W')
            logits = self.head(feats, out_hw)          # (B, K, H, W)
            loss   = self.criterion(logits[:,1,:,:], masks)

        if not torch.isfinite(loss):
            print("Non-finite loss encountered; skipping step.")
            return float("nan"), logits

        if train:
            if self.scaler is not None and self.use_amp:
                self.scaler.scale(loss).backward()
                if self.clip_grad_norm and self.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.clip_grad_norm and self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=self.clip_grad_norm)
                self.optimizer.step()

        return loss.item(), logits

    @torch.no_grad()
    def _colorize_mask(self, m: torch.Tensor, num_classes: int):
        K = num_classes
        B, H, W = m.shape
        out = torch.zeros(B, 3, H, W, device=m.device, dtype=torch.float32)
        if K == 2:
            out[:, :, :, :] = 0.0
            out[:, 0] = (m == 1).float()
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

        imgs_cpu  = imgs[:4].detach().cpu().clamp(0, 1)
        preds_cpu = logits[:4].detach().argmax(1).cpu()
        gts_cpu   = masks[:4].detach().cpu()

        imgs_vis = denorm_imagenet(imgs_cpu).clamp(0,1)

        pred_rgb = self._colorize_mask(preds_cpu, self.cfg["num_classes"])
        gt_rgb   = self._colorize_mask(gts_cpu,   self.cfg["num_classes"])

        H, W = gts_cpu.shape[-2:]
        if imgs_vis.shape[-2:] != (H, W):
            imgs_vis = F.interpolate(imgs_vis, size=(H, W), mode="bilinear", align_corners=False)

        save_image(imgs_vis,  grid_dir / f"{step_tag}_img.png",  nrow=4)
        save_image(pred_rgb,  grid_dir / f"{step_tag}_pred.png", nrow=4)
        save_image(gt_rgb,    grid_dir / f"{step_tag}_gt.png",   nrow=4)

        trip = torch.cat([imgs_vis, pred_rgb, gt_rgb], dim=0)
        grid = make_grid(trip, nrow=4)
        save_image(grid, grid_dir / f"{step_tag}_triptych.png")

    # ---------- MLflow helpers ----------
    def _resolve_mlflow_tracking(self):
        # Prefer env vars; fallback to defaults
        uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not uri:
            # Make a relative file store next to your project root by default
            # (Use an absolute path if you prefer)
            uri = f"file:{(Path.cwd() / 'mlruns').as_posix()}"
        mlflow.set_tracking_uri(uri)

        exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "default")
        mlflow.set_experiment(exp_name)
        return uri, exp_name

    def train(self):
        best_val = float('inf')
        best_path = self.out_dir / "checkpoint_best.pt"
        last_path = self.out_dir / "checkpoint_last.pt"

        tracking_uri, exp_name = self._resolve_mlflow_tracking()

        run_name = f"{self.cfg.get('dino_size','?')}_img{self.cfg['img_size'][0]}_{'lora' if self.cfg.get('use_lora',True) else 'head'}"

        # === Everything MLflow happens here ===
        mlflow.set_experiment(self.cfg.get("mlflow_experiment_name", "default")) 
        with mlflow.start_run(run_name=run_name) as run:
            r = mlflow.active_run()
            assert r is not None, "MLflow run did not start"

            # small canary to force hydration
            mlflow.log_param("run_canary", "ok")
            mlflow.log_metric("canary/step0", 0.0, step=0)

            # Log key params once
            mlflow.log_param("seed",      int(self.seed))
            mlflow.log_param("dino_size", self.cfg.get("dino_size"))
            mlflow.log_param("img_size",  str(self.cfg.get("img_size")))
            mlflow.log_param("use_lora",  bool(self.cfg.get("use_lora", True)))
            mlflow.log_param("lora_rank", int(self.cfg.get("lora_rank", 0)))
            mlflow.log_param("lora_alpha",int(self.cfg.get("lora_alpha", 0)))
            mlflow.log_param("batch_size",int(self.cfg.get("batch_size")))
            mlflow.log_param("epochs",    int(self.cfg.get("epochs")))
            mlflow.log_param("lr",        float(self.cfg.get("lr")))
            mlflow.log_param("weight_decay", float(self.cfg.get("weight_decay")))
            mlflow.log_param("loss", self.cfg.get("loss","ce"))
            mlflow.log_param("class_weights", str(self.cfg.get("class_weights")))
            mlflow.log_param("tversky", f"{self.cfg.get('tversky_alpha',0.7)},{self.cfg.get('tversky_beta',0.3)}")

            # Write run info to disk so you can click back later
            client = MlflowClient()
            art_uri = client.get_run(r.info.run_id).info.artifact_uri
            exp = mlflow.get_experiment(r.info.experiment_id)
            print(f"[mlflow] tracking_uri={mlflow.get_tracking_uri()}  "
                  f"experiment={exp.name}  run_id={r.info.run_id}  artifact_uri={art_uri}")

            with open(self.out_dir / "mlflow_run_id.txt", "w") as f:
                f.write(r.info.run_id + "\n")
                f.write(f"tracking_uri={mlflow.get_tracking_uri()}\n")
                f.write(f"experiment_id={r.info.experiment_id}\n")
                f.write(f"artifact_uri={art_uri}\n")

            tmp_note = self.out_dir / "run_started.txt"
            tmp_note.write_text("trainer started\n")
            mlflow.log_artifact(str(tmp_note), artifact_path="notes")

            # --------- training loop ----------
            for epoch in range(1, self.epochs + 1):
                # TRAIN
                self.backbone.train(False)
                self.head.train(True)

                running = 0.0
                for batch in self.train_loader:
                    loss, _ = self._step(batch, train=True)
                    if isinstance(loss, float) and (loss != loss):  # NaN
                        continue
                    running += loss
                avg_train = running / max(1, len(self.train_loader))

                # VAL
                self.head.eval()
                val_loss = 0.0
                fg_gt_total = fg_pred_total = 0
                bg_gt_total = bg_pred_total = 0

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

                # MLflow scalars
                try:
                    mlflow.log_metric("train/loss", float(avg_train), step=epoch)
                    mlflow.log_metric("val/loss",   float(val_loss),  step=epoch)
                    mlflow.log_metric("val/fg_gt_px",   int(fg_gt_total),   step=epoch)
                    mlflow.log_metric("val/fg_pred_px", int(fg_pred_total), step=epoch)
                    mlflow.log_metric("val/bg_gt_px",   int(bg_gt_total),   step=epoch)
                    mlflow.log_metric("val/bg_pred_px", int(bg_pred_total), step=epoch)
                except Exception as e:
                    print("[mlflow] logging FAILED:", e)

                # Checkpoints
                ckpt = {
                    "head": self.head.state_dict(),
                    "backbone_lora": {k: v for k, v in self.backbone.state_dict().items() if "lora_" in k},
                    "cfg": self.cfg,
                    "epoch": int(epoch),
                    "val_loss": float(val_loss),
                }

                torch.save(ckpt, last_path)
                print(f"[ckpt] wrote {last_path.name}")

                if val_loss < best_val:
                    best_val = float(val_loss)
                    torch.save(ckpt, best_path)
                    print(f"[ckpt] NEW BEST -> {best_path.name} (val_loss={best_val:.4f})")
                    mlflow.log_metric("val/best_loss", best_val, step=epoch)

                # upload previews periodically
                try:
                    if epoch % 5 == 0:
                        mlflow.log_artifacts(str(self.out_dir / "previews"), artifact_path="previews")
                except Exception as e:
                    print("[mlflow] preview artifact upload skipped:", e)

            # end-of-run artifacts
            try:
                mlflow.log_artifacts(str(self.out_dir), artifact_path="run_artifacts")
            except Exception as e:
                print("[mlflow] final artifact upload skipped:", e)
        # context manager will end the run for us
