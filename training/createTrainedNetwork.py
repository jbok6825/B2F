import argparse
import gc
import os

import torch
from process_dataset.Constant import DEVICE, PATH_DB, PATH_MODEL
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from training.CustomDataset import CustomDataset, collate_fn_same_timing
from training.network.Network import Network


def parse_args():
    parser = argparse.ArgumentParser(description="Train Model_Ours")
    parser.add_argument(
        "--dataset_dir",
        default=PATH_DB + "_clipping_random_big",
        help="Processed dataset directory (pickle files).",
    )
    parser.add_argument(
        "--save_dir",
        default=os.path.join(PATH_MODEL, "Model_Ours"),
        help="Directory to save checkpoints.",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--save_step", type=int, default=50, help="Checkpoint save interval (epochs).")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print("device:", device)

    # Ensure global DEVICE used in datasets/utils matches chosen device
    import process_dataset.Constant as C
    C.DEVICE = device

    dataset = CustomDataset(device=device, path_dataset_dir=args.dataset_dir)
    dataloader_1 = DataLoader(
        dataset, args.batch_size, shuffle=True, collate_fn=collate_fn_same_timing, drop_last=True
    )
    dataloader_2 = DataLoader(
        dataset, args.batch_size, shuffle=True, collate_fn=collate_fn_same_timing, drop_last=True
    )

    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter()
    network = create_expanded_network(
        path_preTrainedModel=None,
        use_normalize=True,
        positional_encoding=True,
        style_latent_dim=12 * 16,
        device=device,
    )

    network.train_network(
        dataloader=dataloader_1,
        dataloader_sub=dataloader_2,
        epoch_num=args.epochs,
        writer=writer,
        save_dir=args.save_dir,
        save_step=args.save_step,
        mode="origin",
    )

    del network, writer, dataset, dataloader_1, dataloader_2
    gc.collect()
    torch.cuda.empty_cache()


def create_expanded_network(
    path_preTrainedModel=None,
    use_normalize=True,
    positional_encoding=True,
    style_latent_dim=12 * 16,
    content_vae_mode=False,
    device=DEVICE,
):
    size_fullbody_motion_feature = 144
    size_facial_motion_feature = 53
    face_content_latent_dim = 512
    body_content_latnet = 512

    network = Network(
        device,
        face_dim=size_facial_motion_feature,
        body_dim=size_fullbody_motion_feature,
        face_emotion_latent_dim=style_latent_dim,
        face_content_latent_dim=face_content_latent_dim,
        body_content_latent_dim=body_content_latnet,
        use_normalize=use_normalize,
        vae_mode=True,
        positional_encoding=positional_encoding,
        content_vae_mode=content_vae_mode,
    ).to(DEVICE)

    if path_preTrainedModel is not None:
        pretrained_model = torch.load(path_preTrainedModel)
        network.load_state_dict(pretrained_model.state_dict())

    return network.to(device)


if __name__ == "__main__":
    main()
