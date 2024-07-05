import argparse
import os


def get_idx(filename):
    return int(filename.split('.')[0].split('_')[-1])


def main():
    parser = argparse.ArgumentParser(description='Count instances')
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--start_idx', type=int, help='Start index')
    parser.add_argument('--end_idx', type=int, help='End index')
    args = parser.parse_args()

    labels_dir = os.path.join(args.input_dir, 'labels')
    labels = os.listdir(labels_dir)
    labels = [label for label in labels if label.endswith('.txt')]
    if args.start_idx is not None:
        labels = [
            label for label in labels if args.start_idx <= get_idx(label)
        ]
    if args.end_idx is not None:
        labels = [label for label in labels if get_idx(label) <= args.end_idx]

    num_instances = 0
    for label in labels:
        label_path = os.path.join(labels_dir, label)
        with open(label_path) as f:
            for r in f:
                class_id, x, y, w, h, *instance_id = r.strip().split()
                if not instance_id:
                    num_instances += 1

    print(f'Arguments: {args}')
    print(f'Number of instances: {num_instances}')


if __name__ == '__main__':
    main()
