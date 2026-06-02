# Changelog

## [Unreleased]

### Added
- Add `SparseHybridEncoder` with saliency-based sparse window selection for efficient feature encoding
- Add DINOv3 VisDrone sparse encoder training config and launcher script
- Add TUNING_CKPT environment variable support to experiment launcher
- Add ResNet50 freeze training configurations and scripts for VisDrone experiments
- Add VisDrone validation set inference script with GT comparison visualization
- Add VisDrone test set batch inference script with FPS measurement and multi-experiment summary
- Add encoder-free DINOv3 VisDrone experiment config and launcher under experiments/
- Add DINOv3 architecture documentation with tensor-shape walkthrough

### Changed
- Allow experiments/configs and experiments/scripts to be tracked while keeping generated outputs ignored
- Reduce DINOv3 encoder-free VisDrone experiment total_batch_size from 8 to 6
