#!/usr/bin/env python3

"""VisDrone COCO 标注子集生成工具。

默认用于生成 10%、25%、50% 的训练子集，供 DINOv3/ResNet 小样本实验复用。
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_coco_annotations(json_path):
    with open(json_path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def stratified_split_by_category(annotations, ratio, seed=42):
    random.seed(seed)
    cat_to_annotations = defaultdict(list)
    for annotation in annotations:
        cat_to_annotations[annotation['category_id']].append(annotation)

    subset_annotations = []
    selected_image_ids = set()

    if ratio < 100:
        for ann_list in cat_to_annotations.values():
            if ann_list:
                subset_annotations.append(ann_list[0])
                selected_image_ids.add(ann_list[0]['image_id'])

    remaining_annotations = []
    for ann_list in cat_to_annotations.values():
        remaining_annotations.extend(ann_list[1:])

    num_to_select = int(len(remaining_annotations) * ratio / 100)
    chosen_annotations = random.sample(
        remaining_annotations,
        min(num_to_select, len(remaining_annotations)),
    )

    subset_annotations.extend(chosen_annotations)
    selected_image_ids.update(ann['image_id'] for ann in chosen_annotations)
    return subset_annotations, selected_image_ids


def create_subset_coco(original_data, subset_annotations, selected_image_ids, output_path):
    subset_images = [
        image for image in original_data['images']
        if image['id'] in selected_image_ids
    ]
    subset_data = {
        'info': original_data.get('info', {}),
        'licenses': original_data.get('licenses', []),
        'images': subset_images,
        'annotations': subset_annotations,
        'categories': original_data.get('categories', []),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(subset_data, handle, indent=2)

    return len(subset_images), len(subset_annotations)


def compute_statistics(annotations, categories):
    category_to_count = defaultdict(int)
    for annotation in annotations:
        category_to_count[annotation['category_id']] += 1

    category_names = {category['id']: category['name'] for category in categories}
    rows = []
    for category_id, count in sorted(category_to_count.items()):
        rows.append((category_names.get(category_id, str(category_id)), count))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description='Generate VisDrone COCO subsets')
    parser.add_argument('--input-json', required=True, help='原始 train_coco.json 路径')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--ratios', nargs='+', type=int, default=[10, 25, 50], help='子集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--stratified', action='store_true', help='按类别分层抽样')
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_json)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f'Input annotation file not found: {input_path}')

    data = load_coco_annotations(input_path)
    num_images = len(data['images'])
    num_annotations = len(data['annotations'])

    print(f'Loaded: {input_path}')
    print(f'Images: {num_images}')
    print(f'Annotations: {num_annotations}')

    for ratio in args.ratios:
        if ratio <= 0 or ratio > 100:
            raise ValueError(f'Ratio must be in (0, 100], got {ratio}')

        if args.stratified:
            subset_annotations, selected_image_ids = stratified_split_by_category(
                data['annotations'], ratio, args.seed
            )
        else:
            random.seed(args.seed)
            total = len(data['annotations'])
            chosen = random.sample(data['annotations'], int(total * ratio / 100))
            subset_annotations = chosen
            selected_image_ids = {annotation['image_id'] for annotation in chosen}

        output_path = output_dir / f'train_coco_{ratio}pct.json'
        subset_image_count, subset_annotation_count = create_subset_coco(
            data,
            subset_annotations,
            selected_image_ids,
            output_path,
        )

        print(f'Created: {output_path}')
        print(f'  Image count: {subset_image_count} ({subset_image_count / num_images * 100:.1f}%)')
        print(f'  Annotation count: {subset_annotation_count} ({subset_annotation_count / num_annotations * 100:.1f}%)')

        for category_name, count in compute_statistics(subset_annotations, data.get('categories', []))[:5]:
            print(f'  {category_name}: {count}')


if __name__ == '__main__':
    main()