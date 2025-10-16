from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def _list_files(root: Path, recursive: bool):
    pattern = "**/*" if recursive else "*"
    return sorted([p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in IMG_EXTS])

class PairedDirsSegDataset(Dataset):
    """
    Pair images and masks from two folders.

    Modes:
      - pair_mode="stem"   : match by filename stem (optionally remove mask_prefix/suffix)
      - pair_mode="sorted" : sort both lists and pair by index (requires same counts)

    Extras:
      - mask_prefix / mask_suffix for mask filenames like 'foo_mask.png' or 'gt_foo.png'
      - recursive=True to scan subfolders
      - binarize (0/255 → 0/1) or keep label ids

    Output:
      - image: Tensor CHW (after transform if provided)
      - mask : LongTensor HW (values in {0..K-1} or {0,1} if binarize)
    """
    def __init__(
        self,
        image_dir,
        mask_dir,
        img_size=(308, 308),
        to_rgb=True,
        transform=None,
        binarize=False,
        binarize_threshold=128,
        pair_mode="stem",
        mask_prefix="",
        mask_suffix="",
        recursive=False,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.img_size  = tuple(img_size)
        self.to_rgb    = bool(to_rgb)
        self.transform = transform
        self.binarize  = bool(binarize)
        self.thresh    = int(binarize_threshold)

        imgs  = _list_files(self.image_dir, recursive)
        masks = _list_files(self.mask_dir,  recursive)

        self.pairs = []

        # Check that their are same number of images and masks
        if pair_mode == "sorted":
            if len(imgs) != len(masks):
                raise RuntimeError(
                    f"[pair_mode=sorted] Count mismatch: images={len(imgs)} masks={len(masks)}\n"
                    f"  images={self.image_dir}\n  masks={self.mask_dir}"
                )
            self.pairs = list(zip(imgs, masks))

        else:  # "stem" matching (with the name file)
            def norm_mask_stem(p: Path):
                s = p.stem
                if mask_prefix and s.startswith(mask_prefix):
                    s = s[len(mask_prefix):]
                if mask_suffix and s.endswith(mask_suffix):
                    s = s[: -len(mask_suffix)]
                return s

            mask_map = {norm_mask_stem(mp): mp for mp in masks}

            for ip in imgs:
                mp = mask_map.get(ip.stem, None)
                if mp is not None:
                    self.pairs.append((ip, mp))

        if len(self.pairs) == 0:
            peek_i = [p.name for p in imgs[:6]]
            peek_m = [p.name for p in masks[:6]]
            raise RuntimeError(
                "No (image,mask) pairs found.\n"
                f"  images={self.image_dir}\n"
                f"    first files: {peek_i}\n"
                f"  masks={self.mask_dir}\n"
                f"    first files: {peek_m}\n"
                "Try pair_mode='sorted' or set mask_prefix/mask_suffix/recursive."
            )

    def __len__(self):
        return len(self.pairs)

    def _load_rgb(self, p: Path):
        img = Image.open(p)
        # if TIFF has multiple frames, take the first
        try:
            if getattr(img, "n_frames", 1) > 1:
                img.seek(0)
        except Exception:
            pass
        if self.to_rgb:
            if img.mode == "RGB":
                return img
            # typical EM is grayscale; replicate to RGB
            g = img.convert("L")
            return Image.merge("RGB", (g, g, g))
        else:
            return img

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]

        # image
        img = self._load_rgb(ip).resize(self.img_size, Image.BICUBIC)

        # --- MASK: force single channel ---
        mask = Image.open(mp)
        try:
            if getattr(mask, "n_frames", 1) > 1:
                mask.seek(0)
        except Exception:
            pass

        # Convert to 8-bit single-channel. This handles RGB/RGBA/Palette masks robustly.
        if mask.mode not in ("L", "I;16", "I"):
            try:
                mask = mask.convert("L")
            except Exception:
                # fallback: take first channel if convert fails (e.g., unusual mode)
                mask = mask.split()[0]
        mask = mask.resize(self.img_size, Image.NEAREST)

        # transforms
        if self.transform:
            img = self.transform(img)

        # to numpy (single channel)
        mask = np.array(mask)
        if mask.ndim == 3:       # just in case something slipped through
            mask = mask[..., 0]

        # binarize or keep labels
        if self.binarize:
            mask = (mask > self.thresh).astype(np.int64)  # 0/1
        else:
            mask = mask.astype(np.int64)

        return img, torch.from_numpy(mask)

