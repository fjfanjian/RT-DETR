"""
VisDrone 测试集批量推理脚本
遍历所有实验，对 test_coco.json 中的图片进行推理、测速、保存可视化结果

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/batch_infer_visdrone_test.py [--device cuda]
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig

# VisDrone 类别名称
VISDRONE_CATEGORIES = {
    1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van',
    6: 'truck', 7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor'
}
COLORS = [
    '#FF3838', '#FF9D00', '#FFC926', '#00BAFF', '#00E5FF',
    '#00D7FF', '#AD00FF', '#FF00A0', '#7BFF63', '#00FF9D',
]


# 字体
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except:
    try:
        FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        FONT = ImageFont.load_default()


def load_model(config_path, resume_path, device):
    """加载模型"""
    cfg = YAMLConfig(config_path, resume=resume_path)
    checkpoint = torch.load(resume_path, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    elif 'model' in checkpoint:
        state = checkpoint['model']
    else:
        state = checkpoint
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(device)
    model.eval()
    return model


def load_test_images(ann_file, img_folder):
    """从 COCO JSON 加载测试图片路径列表"""
    with open(ann_file) as f:
        data = json.load(f)
    image_infos = []
    for img in data['images']:
        img_path = os.path.join(img_folder, img['file_name'])
        image_infos.append({
            'id': img['id'],
            'file_name': img['file_name'],
            'path': img_path,
            'width': img.get('width', 0),
            'height': img.get('height', 0),
        })
    return image_infos


def draw_detections(image, boxes, labels, scores, thrh=0.4):
    """在图片上绘制检测框"""
    draw = ImageDraw.Draw(image)
    keep = scores > thrh
    for i in np.where(keep)[0]:
        x1, y1, x2, y2 = map(int, boxes[i])
        label = int(labels[i])
        score = scores[i]
        category = VISDRONE_CATEGORIES.get(label, f'c{label}')
        color = COLORS[label % len(COLORS)]
        text = f'{category} {score:.2f}'
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        bbox = draw.textbbox((x1, y1), text, font=FONT)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1), text, fill='white', font=FONT)
    return image


def inference_single(model, im_pil, device):
    """对单张图片进行推理"""
    w, h = im_pil.size
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    orig_size = torch.tensor([w, h])[None].to(device)
    im_data = transform(im_pil)[None].to(device)
    with torch.no_grad():
        with autocast():
            output = model(im_data, orig_size)
    labels, boxes, scores = output
    labels = labels.cpu().numpy()[0]
    boxes = boxes.cpu().numpy()[0]
    scores = scores.cpu().numpy()[0]
    return labels, boxes, scores


def measure_fps(model, image_infos, device, num_warmup=50, num_measure=500):
    """测量模型推理速度 (FPS)"""
    transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    # 准备一批图片（用数据集中前 N 张）
    sample_infos = image_infos[:max(num_warmup + num_measure, len(image_infos))]
    images_data = []
    orig_sizes = []
    for info in sample_infos:
        if not os.path.exists(info['path']):
            continue
        im_pil = Image.open(info['path']).convert('RGB')
        w, h = im_pil.size
        orig_sizes.append([w, h])
        images_data.append(transform(im_pil))
        if len(images_data) >= num_warmup + num_measure:
            break

    if len(images_data) < 10:
        return 0.0, 0.0, 0

    # warmup
    for i in range(min(num_warmup, len(images_data))):
        im_data = images_data[i][None].to(device)
        orig_size = torch.tensor(orig_sizes[i])[None].to(device)
        with torch.no_grad():
            with autocast():
                _ = model(im_data, orig_size)

    # 测速
    torch.cuda.synchronize()
    start = time.time()
    n_measure = min(num_measure, len(images_data) - num_warmup)
    for i in range(num_warmup, num_warmup + n_measure):
        im_data = images_data[i][None].to(device)
        orig_size = torch.tensor(orig_sizes[i])[None].to(device)
        with torch.no_grad():
            with autocast():
                _ = model(im_data, orig_size)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    fps = n_measure / elapsed
    avg_ms = elapsed / n_measure * 1000
    return fps, avg_ms, n_measure


# ===== 实验定义 =====
EXPERIMENTS = [
    {
        'name': 'dinov3_freeze_10pct',
        'config': 'configs/rtdetrv2/rtdetrv2_dinov3b_visdrone.yml',
    },
    {
        'name': 'dinov3_freeze_25pct',
        'config': 'configs/rtdetrv2/rtdetrv2_dinov3b_visdrone.yml',
    },
    {
        'name': 'dinov3_freeze_full',
        'config': 'configs/rtdetrv2/rtdetrv2_dinov3b_visdrone.yml',
    },
    {
        'name': 'r50_25pct',
        'config': 'configs/rtdetrv2/rtdetrv2_r50vd_visdrone.yml',
    },
    {
        'name': 'r50_freeze_10pct',
        'config': 'configs/rtdetrv2/rtdetrv2_r50vd_visdrone.yml',
    },
    {
        'name': 'r50_freeze_25pct',
        'config': 'configs/rtdetrv2/rtdetrv2_r50vd_visdrone.yml',
    },
    {
        'name': 'r50_freeze_full',
        'config': 'configs/rtdetrv2/rtdetrv2_r50vd_visdrone.yml',
    },
    {
        'name': 'r50_full',
        'config': 'configs/rtdetrv2/rtdetrv2_r50vd_visdrone.yml',
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--experiments-dir', type=str,
                        default='output/experiments')
    parser.add_argument('--out-dir', type=str,
                        default='output/inference_visdrone')
    parser.add_argument('--img-folder', type=str,
                        default='/home/fj/datasets/visdrone/VisDrone2019-DET-train/images')
    parser.add_argument('--ann-file', type=str,
                        default='/home/fj/datasets/visdrone/test_coco.json')
    parser.add_argument('--thrh', type=float, default=0.4,
                        help='置信度阈值')
    parser.add_argument('--max-images', type=int, default=0,
                        help='每张模型最大推理图片数（0=全部）')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # 加载测试图片列表
    image_infos = load_test_images(args.ann_file, args.img_folder)
    print(f'Loaded {len(image_infos)} test images from {args.ann_file}')

    if args.max_images > 0:
        image_infos = image_infos[:args.max_images]

    # 记录总体结果
    summary = []

    for exp in EXPERIMENTS:
        exp_name = exp['name']
        config_path = exp['config']
        resume_path = os.path.join(args.experiments_dir, exp_name, 'best.pth')

        if not os.path.exists(resume_path):
            print(f'\n[SKIP] {exp_name}: best.pth not found at {resume_path}')
            continue

        print(f'\n{"="*60}')
        print(f'[Experiment] {exp_name}')
        print(f'  Config: {config_path}')
        print(f'  Resume: {resume_path}')
        print(f'{"="*60}')

        # 输出目录
        exp_out_dir = os.path.join(args.out_dir, exp_name)
        vis_dir = os.path.join(exp_out_dir, 'visualizations')
        det_dir = os.path.join(exp_out_dir, 'detections')
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(det_dir, exist_ok=True)

        # 加载模型
        print('  Loading model...')
        t0 = time.time()
        model = load_model(config_path, resume_path, device)
        print(f'  Model loaded in {time.time()-t0:.1f}s')

        # 测速
        print('  Measuring FPS...')
        fps, avg_ms, n_meas = measure_fps(model, image_infos, device)
        print(f'  Speed: {fps:.1f} FPS ({avg_ms:.1f} ms per image, measured on {n_meas} images)')

        # 批量推理
        print(f'  Running inference on {len(image_infos)} images...')
        results = []  # COCO 格式结果
        t_start = time.time()

        for idx, info in enumerate(image_infos):
            img_path = info['path']
            if not os.path.exists(img_path):
                print(f'    WARNING: {img_path} not found, skipping.')
                continue

            im_pil = Image.open(img_path).convert('RGB')
            labels, boxes, scores = inference_single(model, im_pil, device)

            # 收集 COCO 格式结果
            keep = scores > args.thrh
            for i in np.where(keep)[0]:
                x1, y1, x2, y2 = boxes[i]
                # COCO format: [x, y, w, h]
                results.append({
                    'image_id': info['id'],
                    'category_id': int(labels[i]),
                    'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    'score': float(scores[i]),
                })

            # 保存可视化（每张图保存带框的结果）
            vis_img = draw_detections(im_pil.copy(), boxes, labels, scores, args.thrh)
            vis_path = os.path.join(vis_dir, info['file_name'])
            vis_img.save(vis_path, quality=95)

            if (idx + 1) % 100 == 0:
                elapsed = time.time() - t_start
                print(f'    [{idx+1}/{len(image_infos)}] {elapsed/(idx+1):.2f}s per image')

        elapsed_total = time.time() - t_start
        actual_fps = len(image_infos) / elapsed_total

        # 保存 COCO 格式检测结果
        det_json_path = os.path.join(det_dir, 'predictions.json')
        with open(det_json_path, 'w') as f:
            json.dump(results, f)
        print(f'  Saved {len(results)} detections to {det_json_path}')

        # 保存测速结果
        speed_info = {
            'experiment': exp_name,
            'num_images': len(image_infos),
            'fps_warmup': round(fps, 1),
            'avg_ms_warmup': round(avg_ms, 1),
            'total_time_sec': round(elapsed_total, 1),
            'fps_actual': round(actual_fps, 1),
            'num_detections': len(results),
        }
        speed_path = os.path.join(exp_out_dir, 'speed.json')
        with open(speed_path, 'w') as f:
            json.dump(speed_info, f, indent=2)

        summary.append(speed_info)
        print(f'  Done! Total: {elapsed_total:.1f}s, Actual FPS: {actual_fps:.1f}')
        print(f'  Visualizations: {vis_dir}')

    # 总体对比
    print(f'\n\n{"="*60}')
    print(f'SUMMARY - All Experiments')
    print(f'{"="*60}')
    print(f'{"Experiment":<25} {"FPS(warm)":<12} {"FPS(real)":<12} {"Detections":<12}')
    print(f'-'*60)
    for s in summary:
        print(f'{s["experiment"]:<25} {s["fps_warmup"]:<12.1f} {s["fps_actual"]:<12.1f} {s["num_detections"]:<12}')
    print(f'-'*60)

    # 写入汇总
    summary_path = os.path.join(args.out_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary saved to {summary_path}')


if __name__ == '__main__':
    main()
