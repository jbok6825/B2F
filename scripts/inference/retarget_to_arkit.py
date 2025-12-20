import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from process_dataset.Constant import ARKIT_BLENDSHAPE  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert generated facial motion (FLAME jaw + expression) to ARKit blendshapes."
    )
    parser.add_argument(
        "--facial_npz",
        required=True,
        help="NPZ produced by scripts/inference/generate_facial_motion.py (contains jaw, expression).",
    )
    parser.add_argument(
        "--flame_to_arkit_model",
        default=str(Path(__file__).resolve().parents[2] / "Model/Model_Ours/model_arkit.pt"),
        help="Path to FLAME->ARKit mapper checkpoint (torch.save(model, ...)).",
    )
    parser.add_argument(
        "--output",
        default="outputs/arkit_blendshapes.npz",
        help="Output NPZ path (stores arkit blendshapes and names).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run conversion on.",
    )
    return parser.parse_args()


def load_facial_npz(path):
    data = np.load(path)
    if "jaw" not in data or "expression" not in data:
        raise ValueError(f"{path} must contain 'jaw' and 'expression' arrays")

    jaw = data["jaw"]
    expr = data["expression"]

    # Allow optional batch dimension
    if expr.ndim == 3:
        expr = expr[0]
    if jaw.ndim == 3:
        jaw = jaw[0]

    if expr.shape[0] != jaw.shape[0]:
        raise ValueError(f"Length mismatch: expression {expr.shape} vs jaw {jaw.shape}")

    return expr, jaw


def load_mapper(model_path, device):
    obj = torch.load(model_path, map_location=device)
    if isinstance(obj, torch.nn.Module):
        model = obj
        # Mirror the Blender loader: push every tensor to the target device
        state = model.state_dict()
        for k, v in state.items():
            state[k] = v.to(device)
        model.load_state_dict(state)
    else:
        raise ValueError(
            "Expected flame_to_arkit_model to be a torch.nn.Module (torch.save(model, ...)). "
            "State-dict-only checkpoints are not supported here."
        )

    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device(args.device)

    expr, jaw = load_facial_npz(args.facial_npz)
    inputs = np.concatenate([expr, jaw], axis=-1)  # expected [T, 103]

    # The ARKit mapper was trained on 100 FLAME expression dims + 3 jaw dims (total 103).
    # If the input comes from the 53-dim B2F output (50 expr + 3 jaw), pad missing expr slots.
    if inputs.shape[1] == 53:
        expr_dim = inputs.shape[1] - 3
        expr_part = inputs[:, :expr_dim]
        jaw_part = inputs[:, -3:]
        padded_expr = np.zeros((inputs.shape[0], 100), dtype=inputs.dtype)
        padded_expr[:, :expr_dim] = expr_part
        inputs = np.concatenate([padded_expr, jaw_part], axis=-1)
    elif inputs.shape[1] != 103:
        raise ValueError(f"Unexpected input dim {inputs.shape[1]}; expected 53 (will pad) or 103.")

    model = load_mapper(args.flame_to_arkit_model, device)

    with torch.no_grad():
        arkit = model(torch.from_numpy(inputs).float().to(device)).cpu().numpy()

    if arkit.shape[1] != len(ARKIT_BLENDSHAPE):
        print(
            f"Warning: output dim {arkit.shape[1]} differs from ARKit blendshape count {len(ARKIT_BLENDSHAPE)}"
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(arkit)
    np.savez(args.output, arkit=arkit, names=np.array(ARKIT_BLENDSHAPE))
    print(f"Saved ARKit blendshapes to {args.output} (shape {arkit.shape})")


if __name__ == "__main__":
    main()
