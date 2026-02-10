#!/usr/bin/env python3
"""Compose Drosophila + Lucchi++ into one dataset.

Writes:
  <OUT_ROOT>/{train,test}/{images,masks}/ and mapping.csv

Example (local; absolute paths are user-specific):
  python scripts/data/compose_em_datasets.py
"""

from pathlib import Path
import csv, shutil, random

# =========================
# USER SETTINGS (edit here; absolute paths are user-specific)
# =========================
BASE = Path("/Users/cfuste/Documents/Data/ElectronMicroscopy")

# Drosophila stack1
DROSO_IMG = BASE / "groundtruth-drosophila-vnc-master/stack1/raw"
DROSO_MSK = BASE / "groundtruth-drosophila-vnc-master/stack1/mitochondria"
DROSO_TRAIN_RATIO = 0.85
SEED = 42

# Lucchi++ root (contains Train_In, Train_Out, Test_In, Test_Out)
LUCCHI_ROOT = BASE / "Lucchi++"

# Output dataset
OUT_ROOT = BASE / "composed-dinopeft"

# How to stage files: "symlink" (recommended) or "copy"
STAGE_MODE = "symlink"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# =========================
# helpers
# =========================
def list_images_recursive(d: Path):
    return sorted([p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])

def pair_by_stem(imgs, masks):
    m = {p.stem: p for p in masks}
    return [(ip, m[ip.stem]) for ip in imgs if ip.stem in m]

def pair_by_sorted(imgs, masks):
    if len(imgs) != len(masks):
        raise RuntimeError(f"sorted pairing needs equal counts, got images={len(imgs)} masks={len(masks)}")
    return list(zip(imgs, masks))

def ensure_out_dirs(root: Path):
    for split in ("train","test"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "masks").mkdir(parents=True, exist_ok=True)

def stage(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if STAGE_MODE == "copy":
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())

def stage_pairs(pairs, out_split_dir: Path, tag: str, split: str, start_idx: int, mapping_rows, counts):
    k = start_idx
    out_img = out_split_dir / "images"
    out_msk = out_split_dir / "masks"
    for ip, mp in pairs:
        new_stem = f"{tag}-{split}-{k:06d}"
        new_img = out_img / f"{new_stem}{ip.suffix.lower()}"
        new_msk = out_msk / f"{new_stem}{mp.suffix.lower()}"
        stage(ip, new_img)
        stage(mp, new_msk)
        mapping_rows.append([
            tag, split, new_stem,
            str(new_img.relative_to(OUT_ROOT)),
            str(new_msk.relative_to(OUT_ROOT)),
            str(ip.resolve()), str(mp.resolve())
        ])
        counts[(tag, split)] = counts.get((tag, split), 0) + 1
        k += 1
    return k

# =========================
# main
# =========================
def main():
    # sanity paths
    for p in [DROSO_IMG, DROSO_MSK, LUCCHI_ROOT]:
        if not p.exists():
            raise SystemExit(f"Missing path: {p}")

    ensure_out_dirs(OUT_ROOT)
    mapping_csv = OUT_ROOT / "mapping.csv"
    mapping_rows = [["dataset","split","new_stem","new_image_rel","new_mask_rel","src_image_abs","src_mask_abs"]]
    counts = {}

    # ---------- Drosophila (85/15 split) ----------
    di = list_images_recursive(DROSO_IMG)
    dm = list_images_recursive(DROSO_MSK)
    if not di or not dm:
        raise SystemExit("Droso: no images or masks found (check paths/extensions).")

    droso_pairs = pair_by_stem(di, dm)
    if not droso_pairs:
        # fallback if stems mismatch
        droso_pairs = pair_by_sorted(di, dm)

    random.Random(SEED).shuffle(droso_pairs)
    n_total = len(droso_pairs)
    n_train = int(round(DROSO_TRAIN_RATIO * n_total))
    droso_train = droso_pairs[:n_train]
    droso_test  = droso_pairs[n_train:]

    _ = stage_pairs(droso_train, OUT_ROOT / "train", "droso", "train", start_idx=0, mapping_rows=mapping_rows, counts=counts)
    _ = stage_pairs(droso_test,  OUT_ROOT / "test",  "droso", "test",  start_idx=0, mapping_rows=mapping_rows, counts=counts)

    # ---------- Lucchi++ (respect its Train/Test) ----------
    train_in  = LUCCHI_ROOT / "Train_In"
    train_out = LUCCHI_ROOT / "Train_Out"
    test_in   = LUCCHI_ROOT / "Test_In"
    test_out  = LUCCHI_ROOT / "Test_Out"
    for p in [train_in, train_out, test_in, test_out]:
        if not p.exists():
            raise SystemExit(f"Lucchi++ path missing: {p}")

    li_tr = list_images_recursive(train_in)
    lm_tr = list_images_recursive(train_out)
    li_te = list_images_recursive(test_in)
    lm_te = list_images_recursive(test_out)

    if not li_tr or not lm_tr:
        print("WARNING: Lucchi Train_* empty or unmatched; skipping Lucchi train.")
    else:
        lucchi_train = pair_by_stem(li_tr, lm_tr) or pair_by_sorted(li_tr, lm_tr)
        _ = stage_pairs(lucchi_train, OUT_ROOT / "train", "lucchi", "train", start_idx=0, mapping_rows=mapping_rows, counts=counts)

    if not li_te or not lm_te:
        print("WARNING: Lucchi Test_* empty or unmatched; skipping Lucchi test.")
    else:
        lucchi_test = pair_by_stem(li_te, lm_te) or pair_by_sorted(li_te, lm_te)
        _ = stage_pairs(lucchi_test, OUT_ROOT / "test", "lucchi", "test", start_idx=0, mapping_rows=mapping_rows, counts=counts)

    # ---------- write mapping ----------
    with mapping_csv.open("w", newline="") as f:
        csv.writer(f).writerows(mapping_rows)

    # ---------- summary ----------
    train_imgs = len(list((OUT_ROOT / "train" / "images").iterdir()))
    test_imgs  = len(list((OUT_ROOT / "test" / "images").iterdir()))
    print("Composed dataset:", OUT_ROOT)
    print(f"  train images: {train_imgs} | droso={counts.get(('droso','train'),0)}  lucchi={counts.get(('lucchi','train'),0)}")
    print(f"  test  images: {test_imgs}  | droso={counts.get(('droso','test'),0)}   lucchi={counts.get(('lucchi','test'),0)}")
    print("Mapping CSV:", mapping_csv)
    if STAGE_MODE == "symlink":
        print("Note: symlinks created. If a tool can't follow symlinks, set STAGE_MODE='copy' and rerun.")

if __name__ == "__main__":
    main()
