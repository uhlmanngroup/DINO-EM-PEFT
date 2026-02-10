# Scripts
Entry points for training, evaluation, analysis, and utilities. Each script includes a usage example at the top of the file.

Absolute paths in some examples are user-specific. Update them for your environment if needed.

## Training And Evaluation
- `scripts/train_em_seg.py`: train segmentation from a YAML config
- `scripts/eval_em_seg.py`: evaluate a trained run and write metrics/figs

## Analysis And Reporting
- `scripts/analysis/summarize_seg_results.py`: aggregate run metrics into CSV/JSON
- `scripts/analysis/plot_seg_summary.py`: generate summary plots from aggregates
- `scripts/analysis/compare_ood_runs.py`: compare multiple OOD runs
- `scripts/analysis/fancy_plot.py`: generate the Lucchi++ comparison figure
- `scripts/run_domain_analysis.py`: compute domain metrics on feature files
- `scripts/run_pca.py`: PCA/UMAP on extracted features

## Features And OOD
- `scripts/extract_features.py`: extract DINO features to NPZ
- `scripts/ood_detection.py`: Mahalanobis OOD detection on embeddings

## Inspection And Smoke Tests
- `scripts/inspect_backbone.py`: module tree and LoRA target inspection
- `scripts/smoke_test.py`: lightweight forward pass for backbones
- `scripts/smoke_test_lora.py`: LoRA injection smoke test

## Data And Utilities
- `scripts/data/compose_em_datasets.py`: build the paired Lucchi++/Droso dataset layout
- `scripts/utils/move_result_logs.py`: backfill Slurm logs into run dirs
- `scripts/utils/param_count_table.py`: parameter count table for backbones and LoRA

## Ablations
- `scripts/ablation_paired_balance_dinov2.py`: balanced vs unbalanced paired ablation
