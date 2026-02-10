#!/usr/bin/env python3
"""Train EM segmentation with DINO backbones.

Example (local):
    python scripts/train_em_seg.py --cfg configs/mac/lucchi_dinov2_lora_mac.yaml

Example (cluster):
    sbatch slurm/single_lucchi_dinov2.sbatch configs/cluster/lucchi_dinov2_cluster.yaml
"""

import argparse
from dino_peft.trainers.seg_trainer import SegTrainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cfg",
        "--config",
        dest="cfg",
        type=str,
        default="configs/mac/lucchi_dinov2_lora_mac.yaml",
        help="Path to YAML config.",
    )
    args = ap.parse_args()

    trainer = SegTrainer(args.cfg)
    trainer.train()

if __name__ == "__main__":
    main()
