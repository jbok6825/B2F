import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
from CharacterAnimationTools.anim import amass, bvh
from CharacterAnimationTools.util import quat

from process_dataset.Constant import PATH_MODEL, DATASET_EXTRACT_JOINT_LIST, RUNTIME_EXTRACT_JOINT_LIST
from process_dataset import utils
from training.CustomDataset import CustomDataset
from training.createTrainedB2FNetwork import create_expanded_network


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate facial motion (jaw + expression) using Model_Ours."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--dataset_dir",
        help="Path to processed dataset directory (pickle files) created via scripts/dataset tools.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index (with --dataset_dir).",
    )
    src.add_argument(
        "--body_npy",
        help="Raw Motion-X body npy path (SMPLX parameters).",
    )
    src.add_argument(
        "--body_bvh",
        help="Retargeted BVH path containing the required joints.",
    )
    parser.add_argument(
        "--style_npy",
        help="Style Motion-X npy path (required with --body_npy/--body_bvh).",
    )
    parser.add_argument(
        "--model",
        default=os.path.join(PATH_MODEL, "Model_Ours/model_ours.pth"),
        help="Path to the trained Model_Ours checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="outputs/facial_motion.npz",
        help="Output npz path to save jaw/expression arrays.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: auto).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    if args.dataset_dir:
        sample = _load_from_dataset(args.dataset_dir, args.index, device)
        body = sample["fullbody_feature"].unsqueeze(0).to(device)
        style = sample["facial_style_feature"].unsqueeze(0).to(device)
    else:
        body_motion = _load_body_motion(args, device)
        style = _load_style_feature(args.style_npy, device)
        body = body_motion

    network = create_expanded_network(path_preTrainedModel=args.model).to(device)
    network.eval()

    with torch.no_grad():
        output = network(
            {"body_motion_content": body, "facial_motion_style": style},
            is_runtime=True,
        )
    blendshape = output["blendshape_output"][0]  # [T, 53]
    jaw = blendshape[:, :3].cpu().numpy()
    expression = blendshape[:, 3:].cpu().numpy()
 
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output, jaw=jaw, expression=expression)
    print(f"Saved facial motion to {args.output} (jaw shape {jaw.shape}, expr shape {expression.shape})")


def _load_from_dataset(dataset_dir, index, device):
    dataset = CustomDataset(device=device, path_dataset_dir=dataset_dir)
    if index < 0 or index >= len(dataset):
        raise IndexError(f"index {index} out of range for dataset of length {len(dataset)}")
    return dataset[index]


def _load_style_feature(style_npy, device):
    if not style_npy:
        raise ValueError("style_npy is required when using body_npy/body_bvh")
    motion_style = np.load(style_npy)
    motion_parms_style = utils.parameterize_motionX(motion_style)
    facial_feature_style = utils.get_facial_feature(motion_parms_style)
    facial_style_feature = torch.cat(
        (facial_feature_style["jaw"], facial_feature_style["face_expr"]), dim=-1
    ).unsqueeze(0)
    return facial_style_feature.to(device)


def _load_body_motion(args, device):
    if args.body_npy:
        motion = np.load(args.body_npy)
        smplh = utils.motionX2smplh(motion)
        anim = amass.load(
            amass_motion_file=smplh,
            remove_betas=True,
            gender="neutral",
            anim_name=os.path.basename(args.body_npy),
            load_hand=True,
        )
    elif args.body_bvh:
        anim = bvh.load(filepath=args.body_bvh)
    else:
        raise ValueError("Provide either --body_npy or --body_bvh")

    extract_idx = [anim.joint_names.index(j) for j in DATASET_EXTRACT_JOINT_LIST]
    proj_root_pos = torch.tensor(anim.proj_root_pos(), device=device, dtype=torch.float32) / 100
    proj_root_rot = torch.tensor(quat.to_xform(anim.proj_root_rot), device=device, dtype=torch.float32)
    character_local_coordinate = utils.get_character_local_coordinate_from_projeted_info(
        proj_root_pos, proj_root_rot
    )

    global_position = torch.tensor(anim.gpos, device=device)[:, extract_idx, :] / 100
    global_orientation = torch.tensor(quat.to_xform(anim.grot)[:, extract_idx, :], device=device)
    global_velocity = torch.tensor(anim.gposvel[:, extract_idx, :], device=device) / 100

    (
        current_position,
        current_orientation,
        current_velocity,
    ) = utils.get_motion_feature(
        global_position, global_orientation, global_velocity, character_local_coordinate, only_current=True
    )

    fullbody_feature = utils.get_formatted_data(
        position_feature=current_position,
        orientation_feature=current_orientation,
        velocity_feature=current_velocity,
        face_expr_style_feature=None,
        jaw_style_feature=None,
        style_code=None,
    )["fullbody_feature"].unsqueeze(0)

    return fullbody_feature.to(device)


if __name__ == "__main__":
    main()
