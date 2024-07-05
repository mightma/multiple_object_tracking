import argparse
import glob
import os
import os.path as osp
import shutil

import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Preprocess dataset')
    parser.add_argument('--input_dir',
                        type=str,
                        required=True,
                        help='Input directory')
    parser.add_argument('--output_dir',
                        type=str,
                        required=True,
                        help='Output directory')
    parser.add_argument('--override',
                        action='store_true',
                        help='Override existing files')
    parser.add_argument('--area_threshold',
                        type=float,
                        default=0.001,
                        help='Area threshold')
    args = parser.parse_args()

    assert osp.isdir(
        args.input_dir), f"Input directory {args.input_dir} does not exist"
    if osp.isdir(args.output_dir) and not args.override:
        raise ValueError(
            f"Output directory {args.output_dir} already exists, use --override to overwrite"
        )

    target_dir = args.output_dir
    os.makedirs(target_dir, exist_ok=True)

    image_paths = glob.glob(osp.join(args.input_dir, '*.jpg'))

    count_images = 0
    count_boxes = 0
    count_boxes_total = 0
    for image_path in tqdm(image_paths):
        anno_txt_path = osp.splitext(image_path)[0] + '.txt'
        anno_txt_path = osp.join(args.input_dir, osp.basename(anno_txt_path))
        assert osp.isfile(
            anno_txt_path), f"Annotation file {anno_txt_path} does not exist"

        rows = []
        with open(anno_txt_path, 'r') as f:
            for r in f:
                label, x, y, w, h = r.strip().split()
                label = int(label)
                x, y, w, h = map(float, [x, y, w, h])
                area = w * h
                if area >= args.area_threshold:
                    rows.append(r)
                count_boxes_total += 1

        if len(rows) == 0:
            continue

        target_image_path = osp.join(target_dir, osp.basename(image_path))
        if not os.path.exists(target_image_path):
            shutil.copy2(image_path, target_dir)
        target_anno_txt_path = osp.join(target_dir,
                                        osp.basename(anno_txt_path))
        with open(target_anno_txt_path, 'w') as f:
            for r in rows:
                f.write(r)
        count_images += 1
        count_boxes += len(rows)

    print(f"Number of images: {count_images} / {len(image_paths)}")
    print(f"Number of boxes: {count_boxes} / {count_boxes_total}")


if __name__ == '__main__':
    main()
