"""Constructs the dataset annotations for the ZMOT dataset.
The following steps are performed:
1. Split the images into training and validation sets.
2. Copy the images and annotations to the corresponding directories.
3. Statistics the number of classes in the dataset.
4. Convert the annotations to COCO format.
"""
import glob
import json
import os
import os.path as osp
import random
import shutil
from collections import Counter
from typing import Optional
from warnings import warn

from tqdm import tqdm

ZMOT_CLASSES = (
    "道路不洁_烟头、瓜果皮核",
    "道路不洁_白色漂浮物",
    "道路不洁_漂浮物（白色漂浮物+其他漂浮物）",
    "道路不洁_食物包装垃圾",
    "道路不洁_落叶堆积",
    "道路不洁_食物包装垃圾(饮料瓶)",
    "道路不洁_其他漂浮物",
    "道路不洁_打包垃圾袋",
    "道路不洁_油污、水渍",
    "道路不洁_人畜粪便",
    "道路不洁_果皮、蔬菜叶",
    "道路不洁_泥土、沙石、碎石",
    "道路不洁_散乱垃圾",
    "道路不洁_食物包装垃圾(易拉罐)",
    "垃圾箱满溢_果皮箱满溢",
    "垃圾箱满溢_垃圾桶满溢",
    "道路不洁_食物包装垃圾(烟盒)",
)


def convert_to_coco_format(img_paths, data_dir, categories):
    """Converts the annotations of the images to COCO format.
    Use outside categories to ensure the same order of categories
    in training and validation sets.
    """
    images = []
    annotations = []
    for img in img_paths:
        anno_file = osp.splitext(img)[0] + '.json'
        anno_file = osp.join(data_dir, anno_file)
        anno = json.load(open(anno_file))

        id = len(images) + 1
        filename = osp.basename(img)
        width, height = anno['imageWidth'], anno['imageHeight']
        date_captured = None
        images.append({
            "id": id,
            "file_name": filename,
            "width": width,
            "height": height,
            "date_captured": date_captured
        })
        for obj in anno['shapes']:
            # label = obj['label']
            label = 'Rubbish'
            cat_id = categories.index(label)
            x1, y1 = obj['points'][0]
            x2, y2 = obj['points'][1]
            x1, x2, y1, y2 = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
            area = (x2 - x1) * (y2 - y1)
            bbox = [x1, y1, x2 - x1, y2 - y1]
            # bbox = [x1, y1, x2, y2]
            seg = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            iscrowd = 0
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": area,
                "segmentation": seg,
                "iscrowd": iscrowd
            })
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": i,
            "name": cat
        } for i, cat in enumerate(categories)]
    }


def construct_coco_dataset(use_1cls: bool = False):
    data_dir = './zmot/1000_samples/positive'

    # List all images and split them into training and validation sets
    imgs = []
    for name in os.listdir(data_dir):
        prefix, extension = osp.splitext(name)
        if extension == '.jpg':
            anno_file = osp.join(data_dir, prefix + '.json')
            assert osp.exists(
                anno_file
            ), f"Annotation file {prefix}.json not found for image {name}"
            imgs.append(name)

    # Use sort and seed to ensure the predictability of the split
    imgs.sort()
    random.seed(101)
    random.shuffle(imgs)

    num_imgs = len(imgs)
    val_ratio = 0.1
    split_index = int(val_ratio * num_imgs)
    train_imgs = imgs[split_index:]
    val_imgs = imgs[:split_index]
    print(f"Total number of images: {num_imgs}")
    print(f"Number of training images: {len(train_imgs)}")
    print(f"Number of validation images: {len(val_imgs)}")

    # Write the split to JSON files
    train_path = './zmot/1000_samples/train.json'
    val_path = './zmot/1000_samples/val.json'
    print(f"Saving training images to {train_path}")
    with open(train_path, 'w') as f:
        json.dump(train_imgs, f)
    print(f"Saving validation images to {val_path}")
    with open(val_path, 'w') as f:
        json.dump(val_imgs, f)

    train_image_dir = './zmot/1000_samples/train2017'
    val_image_dir = './zmot/1000_samples/val2017'
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)

    # Statistics all categories
    labels = []
    split_labels = {'train': [], 'val': []}
    for img in tqdm(imgs):
        split = 'train' if img in train_imgs else 'val'
        image_path = osp.join(data_dir, img)
        anno_file = osp.join(data_dir, osp.splitext(img)[0] + '.json')
        target_dir = train_image_dir if split == 'train' else val_image_dir
        if not osp.exists(osp.join(target_dir, img)):
            shutil.copy2(image_path, target_dir)
        if not osp.exists(osp.join(target_dir, osp.basename(anno_file))):
            shutil.copy2(anno_file, target_dir)
        anno = json.load(open(anno_file))
        for obj in anno['shapes']:
            label = 'Rubbish' if use_1cls else obj['label']
            labels.append(label)
            split_labels[split].append(label)
    label_counter = Counter(labels)
    print(f"Number of classes: {len(label_counter)}")
    for split in 'train', 'val':
        counter = Counter(split_labels[split])
        print(f"Number of classes in {split} set: {len(counter)}")
        for label, count in sorted(counter.items(),
                                   key=lambda x: x[1],
                                   reverse=True):
            print(f"  * {label}: {count} / {label_counter[label]}")

    classes_path = './zmot/1000_samples/zmot_classes.txt'
    categories = sorted(label_counter.items(),
                        key=lambda x: x[1],
                        reverse=True)
    categories = [cat for cat, _ in categories]
    with open(classes_path, 'w') as f:
        for label in categories:
            f.write(label + '\n')
    print(
        'Please copy the following classes to `YOLOX/yolox/data/datasets/zmot_classes.py`:'
    )
    print('```\nZMOT_CLASSES = (')
    for label in categories:
        print(f'    "{label}",')
    print(')\n```')

    # Convert to COCO format
    train_imgs_coco = convert_to_coco_format(train_imgs, data_dir, categories)
    val_imgs_coco = convert_to_coco_format(val_imgs, data_dir, categories)
    train_coco_path = './zmot/1000_samples/annotations/zmot_train_coco.json'
    val_coco_path = './zmot/1000_samples/annotations/zmot_val_coco.json'
    os.makedirs(osp.dirname(train_coco_path), exist_ok=True)
    print(f"Saving training images in COCO format to {train_coco_path}")
    with open(train_coco_path, 'w') as f:
        json.dump(train_imgs_coco, f)
    print(f"Saving validation images in COCO format to {val_coco_path}")
    with open(val_coco_path, 'w') as f:
        json.dump(val_imgs_coco, f)


def _construct_ultralytics_dataset(data_dir: str,
                                   override: bool = True,
                                   target_dir: Optional[str] = None,
                                   use_1cls: bool = False):
    images = glob.glob(osp.join(data_dir, '*.jpg'))
    if target_dir is None:
        target_dir = data_dir
    else:
        print(f"Copying images and annotations to {target_dir}")
        os.makedirs(target_dir, exist_ok=True)

    class_to_id = {cls: i for i, cls in enumerate(ZMOT_CLASSES)}

    for image_path in tqdm(images):
        anno_path = osp.splitext(image_path)[0] + '.json'
        anno_txt_path = osp.splitext(image_path)[0] + '.txt'
        anno_txt_path = osp.join(target_dir, osp.basename(anno_txt_path))
        if os.path.exists(anno_txt_path) and not override:
            print(
                f"Annotation file {anno_txt_path} already exists for image {image_path}"
            )
            continue
        if not os.path.exists(anno_path):
            warn(
                f"Annotation file {anno_path} not found for image {image_path}"
            )
            continue
        target_image_path = osp.join(target_dir, osp.basename(image_path))
        if not os.path.exists(target_image_path):
            shutil.copy2(image_path, target_dir)

        anno = json.load(open(anno_path))
        width, height = anno['imageWidth'], anno['imageHeight']
        with open(anno_txt_path, 'w') as f:
            for obj in anno['shapes']:
                label = class_to_id[obj['label']] if not use_1cls else 0
                x1, y1 = obj['points'][0]
                x2, y2 = obj['points'][1]
                x1, x2, y1, y2 = min(x1, x2), max(x1,
                                                  x2), min(y1,
                                                           y2), max(y1, y2)
                x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                x, y, w, h = x / width, y / height, w / width, h / height
                if x < 0 or x > 1 or y < 0 or y > 1 or w < 0 or w > 1 or h < 0 or h > 1:
                    print(os.path.basename(image_path))
                    print(
                        f"Bounding box out of image range: {x}, {y}, {w}, {h}")
                    continue
                f.write(f"{label} {x} {y} {w} {h}\n")


def construct_ultralytics_dataset(use_1cls: bool):
    train_data_dir = './zmot/1000_samples/train2017'
    target_train_data_dir = './zmot/1000_samples/train2017_1cls'
    _construct_ultralytics_dataset(train_data_dir,
                                   override=True,
                                   target_dir=target_train_data_dir,
                                   use_1cls=use_1cls)
    val_data_dir = './zmot/1000_samples/val2017'
    target_val_data_dir = './zmot/1000_samples/val2017_1cls'
    _construct_ultralytics_dataset(val_data_dir,
                                   override=True,
                                   target_dir=target_val_data_dir,
                                   use_1cls=use_1cls)


if __name__ == '__main__':
    use_1cls = True  # Use 1 class for all objects
    construct_coco_dataset(use_1cls=use_1cls)
    construct_ultralytics_dataset(use_1cls=use_1cls)
