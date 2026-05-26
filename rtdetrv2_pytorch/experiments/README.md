# RT-DETRv2 VisDrone 实验包

本目录集中放置 DINOv3 + RT-DETRv2 在 VisDrone 上的核心实验配置、启动脚本与使用说明，避免训练入口和研究设计文件分散在仓库不同位置。

## 目录结构

```text
experiments/
├── README.md
├── configs/
│   ├── dinov3_freeze_full.yml
│   ├── dinov3_freeze_10pct.yml
│   ├── dinov3_freeze_25pct.yml
│   ├── dinov3_full_finetune.yml
│   ├── dinov3_freeze_full_encoder_free.yml
│   ├── r50_full.yml
│   └── r50_25pct.yml
├── data/
│   └── visdrone_subsets/
├── scripts/
│   ├── prepare_visdrone_subsets.sh
│   ├── run_train.sh
│   ├── train_dinov3_freeze_full.sh
│   ├── train_dinov3_freeze_10pct.sh
│   ├── train_dinov3_freeze_25pct.sh
│   ├── train_dinov3_full_finetune.sh
│   ├── train_dinov3_freeze_full_encoder_free.sh
│   ├── train_r50_full.sh
│   └── train_r50_25pct.sh
└── tools/
    └── prepare_visdrone_subsets.py
```

## 实验矩阵

| 实验名 | 配置文件 | 启动脚本 | 目的 |
| --- | --- | --- | --- |
| dinov3_freeze_full | experiments/configs/dinov3_freeze_full.yml | experiments/scripts/train_dinov3_freeze_full.sh | DINOv3 冻结主干，全量 VisDrone |
| dinov3_freeze_full_encoder_free | experiments/configs/dinov3_freeze_full_encoder_free.yml | experiments/scripts/train_dinov3_freeze_full_encoder_free.sh | E18a：移除 HybridEncoder 的首轮 encoder-free 对照 |
| dinov3_freeze_10pct | experiments/configs/dinov3_freeze_10pct.yml | experiments/scripts/train_dinov3_freeze_10pct.sh | 小样本 10%，验证迁移优势 |
| dinov3_freeze_25pct | experiments/configs/dinov3_freeze_25pct.yml | experiments/scripts/train_dinov3_freeze_25pct.sh | 小样本 25%，验证标注效率 |
| dinov3_full_finetune | experiments/configs/dinov3_full_finetune.yml | experiments/scripts/train_dinov3_full_finetune.sh | DINOv3 全量微调上限 |
| r50_full | experiments/configs/r50_full.yml | experiments/scripts/train_r50_full.sh | ResNet-50 全量数据基线 |
| r50_25pct | experiments/configs/r50_25pct.yml | experiments/scripts/train_r50_25pct.sh | ResNet-50 小样本对照 |

## 前置条件

1. 已安装 RT-DETRv2 训练环境。
2. VisDrone 数据已整理为以下默认结构：

```text
/home/fj/datasets/visdrone/
├── VisDrone2019-DET-train/images/
├── VisDrone2019-DET-val/images/
├── train_coco.json
└── val_coco.json
```

3. DINOv3 权重默认路径：

```text
/home/fj/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

## 环境变量

所有脚本都支持通过环境变量覆盖默认路径和设备设置：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| CUDA_VISIBLE_DEVICES | 0 | 使用的 GPU 编号 |
| NPROC_PER_NODE | 自动按 GPU 数推断 | torchrun 进程数 |
| VISDRONE_ROOT | /home/fj/datasets/visdrone | VisDrone 根目录 |
| DINOV3_WEIGHTS | /home/fj/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth | DINOv3 权重路径 |
| OUTPUT_ROOT | /home/fj/RT-DETR/rtdetrv2_pytorch/output/experiments | 实验输出根目录 |

## 使用步骤

### 1. 生成小样本子集

```bash
cd /home/fj/RT-DETR/rtdetrv2_pytorch
bash experiments/scripts/prepare_visdrone_subsets.sh
```

默认会生成：

- experiments/data/visdrone_subsets/train_coco_10pct.json
- experiments/data/visdrone_subsets/train_coco_25pct.json
- experiments/data/visdrone_subsets/train_coco_50pct.json

### 2. 启动单个实验

```bash
cd /home/fj/RT-DETR/rtdetrv2_pytorch
bash experiments/scripts/train_dinov3_freeze_full.sh
```

### 3. 覆盖默认路径示例

```bash
cd /home/fj/RT-DETR/rtdetrv2_pytorch
CUDA_VISIBLE_DEVICES=0,1 \
VISDRONE_ROOT=/data/visdrone \
DINOV3_WEIGHTS=/data/weights/dinov3_vitb16.pth \
bash experiments/scripts/train_dinov3_freeze_25pct.sh
```

## 日志与输出

每个实验都会在以下目录下生成日志和权重：

```text
output/experiments/<experiment_name>/
├── train.log
├── config.yml 或运行期配置缓存
├── best.pth
└── last.pth
```

## 推荐执行顺序

1. r50_full
2. dinov3_freeze_full
3. dinov3_freeze_full_encoder_free
4. dinov3_freeze_10pct
5. dinov3_freeze_25pct
6. r50_25pct
7. dinov3_full_finetune

## 说明

- 小样本实验依赖 experiments/data/visdrone_subsets 中的子集标注文件。
- 各 YAML 配置只保留实验差异项，基础训练参数复用仓库已有配置。
- 所有脚本会自动将 VisDrone 路径与 DINOv3 权重路径通过命令行覆盖到运行配置中，降低硬编码路径带来的维护成本。