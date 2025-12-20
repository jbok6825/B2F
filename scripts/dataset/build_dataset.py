import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from process_dataset import Constant


def parse_args():
    parser = argparse.ArgumentParser(description="Build processed dataset (Motion-X) for Model_Ours.")
    parser.add_argument(
        "--data_root",
        required=True,
        help="Root path to Motion-X npy files (sets Constant.PATH_DB_ORIGIN).",
    )
    parser.add_argument(
        "--style_root",
        required=True,
        help="Root path to style label txt files (sets Constant.PATH_DB_STYLE_ORIGIN).",
    )
    parser.add_argument(
        "--bvh_root",
        required=True,
        help="Root path to Motion-X BVH files if available (sets Constant.PATH_DB_BVH_ORIGIN).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for processed pickle files. Default: <repo>/dataset_clipping_random_big",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Override constants before importing dataset creator logic
    Constant.PATH_DB_ORIGIN = args.data_root.rstrip("/")
    Constant.PATH_DB_STYLE_ORIGIN = args.style_root.rstrip("/")
    Constant.PATH_DB_BVH_ORIGIN = args.bvh_root.rstrip("/")
    if args.output_dir:
        Constant.PATH_DB = args.output_dir.rstrip("/")

    # Late import so constants are already overridden
    from process_dataset import DatasetCreater

    os.makedirs(Constant.PATH_DB + "_clipping_random_big", exist_ok=True)
    DatasetCreater.main()


if __name__ == "__main__":
    main()
