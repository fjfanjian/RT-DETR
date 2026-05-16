"""
VisDrone验证集图片推理 + Ground Truth对比可视化
Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/infer_visdrone.py
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import json
import argparse

from src.core import YAMLConfig

# VisDrone 类别名称 (category_id -> name)
VISDRONE_CATEGORIES = {
    1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van',
    6: 'truck', 7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor'
}

# 用于draw文本的字体
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
except:
    try:
        FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
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


def load_ground_truth(ann_file):
    """加载COCO格式的GT标注"""
    with open(ann_file) as f:
        data = json.load(f)

    # 构建 image_id -> file_name 映射
    img_id_to_file = {}
    for img in data['images']:
        img_id_to_file[img['id']] = img['file_name']

    # 构建 image_id -> annotations 映射
    img_id_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)

    return img_id_to_file, img_id_to_anns


def get_gt_for_image(file_name, img_id_to_file, img_id_to_anns):
    """获取指定文件的GT标注"""
    # 找到对应的 image_id
    target_img_id = None
    for img_id, fname in img_id_to_file.items():
        if fname == file_name:
            target_img_id = img_id
            break

    if target_img_id is None:
        return [], []

    anns = img_id_to_anns.get(target_img_id, [])
    boxes = []
    labels = []
    for ann in anns:
        # COCO bbox: [x, y, w, h] -> [x1, y1, x2, y2]
        x, y, w, h = ann['bbox']
        boxes.append([x, y, x + w, y + h])
        labels.append(ann['category_id'])

    return boxes, labels


def draw_boxes(image, boxes, labels, scores=None, color='red', gt=False):
    """在图片上绘制检测框，返回image方便链式调用"""
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        if gt:
            label_text = VISDRONE_CATEGORIES.get(labels[i], f'class{labels[i]}')
            text = f'GT: {label_text}'
        else:
            label_text = VISDRONE_CATEGORIES.get(labels[i], f'class{labels[i]}')
            text = f'{label_text} {scores[i]:.2f}' if scores is not None else label_text

        # 绘制文本背景和文字
        bbox = draw.textbbox((x1, y1), text, font=FONT)
        draw.rectangle(bbox, fill=color)
        text_color = 'white'
        draw.text((x1, y1), text, fill=text_color, font=FONT)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default='configs/rtdetrv2/rtdetrv2_dinov3b_visdrone.yml')
    parser.add_argument('-r', '--resume', type=str,
                        default='output/experiments/dinov3_freeze_full/best.pth')
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('--img-dir', type=str,
                        default='/home/fj/datasets/visdrone/VisDrone2019-DET-val/images')
    parser.add_argument('--ann-file', type=str,
                        default='/home/fj/datasets/visdrone/val_coco.json')
    parser.add_argument('--out-dir', type=str,
                        default='output/inference_visdrone_val')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'results'), exist_ok=True)

    # 指定的12张图片
    image_names = [
        '0000001_02999_d_0000005.jpg',
        '0000021_00000_d_0000001.jpg',
        '0000022_00000_d_0000004.jpg',
        '0000023_00000_d_0000008.jpg',
        '0000024_00000_d_0000012.jpg',
        '0000026_00000_d_0000024.jpg',
        '0000055_00000_d_0000109.jpg',
        '0000069_00001_d_0000001.jpg',
        '0000072_02834_d_0000003.jpg',
        '0000076_02142_d_0000009.jpg',
        '0000081_00000_d_0000001.jpg',
        '0000086_00000_d_0000001.jpg',
    ]

    print(f'Loading model from {args.resume}...')
    model = load_model(args.config, args.resume, device)
    print('Model loaded.')

    # 加载GT
    img_id_to_file, img_id_to_anns = load_ground_truth(args.ann_file)
    print('Ground truth loaded.')

    # 图像预处理
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    for img_name in image_names:
        img_path = os.path.join(args.img_dir, img_name)
        if not os.path.exists(img_path):
            print(f'WARNING: {img_path} not found, skipping.')
            continue

        print(f'\nProcessing {img_name}...')

        # --- 加载图片 ---
        im_pil = Image.open(img_path).convert('RGB')
        w, h = im_pil.size

        # --- Ground Truth ---
        gt_boxes, gt_labels = get_gt_for_image(img_name, img_id_to_file, img_id_to_anns)
        print(f'  GT objects: {len(gt_boxes)}')
        for lb in sorted(set(gt_labels)):
            cnt = gt_labels.count(lb)
            print(f'    {VISDRONE_CATEGORIES[lb]} (class {lb}): {cnt}')

        # --- 推理 ---
        orig_size = torch.tensor([w, h])[None].to(device)
        im_data = transforms(im_pil)[None].to(device)

        with torch.no_grad():
            with autocast():
                output = model(im_data, orig_size)

        labels, boxes, scores = output
        labels = labels.cpu().numpy()[0]
        boxes = boxes.cpu().numpy()[0]
        scores = scores.cpu().numpy()[0]

        # 过滤低置信度
        thrh = 0.4
        keep = scores > thrh
        pred_labels = labels[keep].astype(int)
        pred_boxes = boxes[keep]
        pred_scores = scores[keep]
        print(f'  Predictions (thresh={thrh}): {len(pred_labels)}')
        for lb in sorted(set(pred_labels)):
            cnt = list(pred_labels).count(lb)
            name = VISDRONE_CATEGORIES.get(lb, f'unknown({lb})')
            print(f'    {name} (class {lb}): {cnt}')

        # --- 创建可视化 ---
        # 创建宽图: 左GT右Prediction
        vis_w = w * 2 + 20
        vis_h = h + 40  # 顶部留40px标题
        canvas = Image.new('RGB', (vis_w, vis_h), color='white')

        # 复制原图两份
        gt_img = im_pil.copy()
        pred_img = im_pil.copy()

        # 绘制GT (绿色)
        draw_boxes(gt_img, gt_boxes, gt_labels, color='lime', gt=True)

        # 绘制预测 (红色)
        draw_boxes(pred_img, pred_boxes, pred_labels, scores=pred_scores, color='red', gt=False)

        # 合并
        canvas.paste(gt_img, (0, 40))
        canvas.paste(pred_img, (w + 20, 40))

        # 标题
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 5), f'GT ({img_name})', fill='lime', font=FONT)
        draw.text((w + 30, 5), f'Prediction ({img_name})', fill='red', font=FONT)

        # 图例
        legend_y = vis_h - 20
        draw.rectangle([10, legend_y - 5, 30, legend_y + 5], fill='lime')
        draw.text((35, legend_y - 8), 'Ground Truth', fill='black', font=FONT)
        draw.rectangle([200, legend_y - 5, 220, legend_y + 5], fill='red')
        draw.text((225, legend_y - 8), 'Prediction', fill='black', font=FONT)

        base_name = img_name.replace('.jpg', '')
        out_path = os.path.join(args.out_dir, 'results', f'{base_name}_comparison.jpg')
        canvas.save(out_path, quality=95)
        print(f'  Saved: {out_path}')

        # 同时单独保存GT和Prediction图
        gt_out = os.path.join(args.out_dir, 'results', f'{base_name}_gt.jpg')
        pred_out = os.path.join(args.out_dir, 'results', f'{base_name}_pred.jpg')
        draw_boxes(im_pil.copy(), gt_boxes, gt_labels, color='lime', gt=True).save(gt_out, quality=95)
        # 重新加载绘制预测
        pred_img_only = im_pil.copy()
        draw_boxes(pred_img_only, pred_boxes, pred_labels, scores=pred_scores, color='red', gt=False)
        pred_img_only.save(pred_out, quality=95)

    print(f'\nAll done! Results saved to {args.out_dir}/results/')


if __name__ == '__main__':
    main()
