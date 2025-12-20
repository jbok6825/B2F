import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from process_dataset.Constant import PATH_MODEL
from training.CustomDataset_B2F import CustomDataset
from training.network_flame_to_arkit.Network_MoE import Network

def main():
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print("device:", DEVICE)


    batch_size = 512
    dir_model_save = os.path.join(PATH_MODEL, "Model_Ours")
    os.makedirs(dir_model_save, exist_ok=True)

    network = Network()
    network.to(DEVICE)
    dataset = CustomDataset()
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    writer = SummaryWriter()

    epoch_num = 1000
    network.train_network(
        dataloader=dataloader,
        epoch_num=epoch_num,
        writer=writer,
        save_dir=dir_model_save,
        save_step=50,
    )

    # Convenience: copy the final checkpoint to the default inference name
    final_ckpt = os.path.join(dir_model_save, f"model_{epoch_num}epoch.pt")
    default_ckpt = os.path.join(dir_model_save, "model_arkit.pt")
    if os.path.exists(final_ckpt):
        try:
            import shutil
            shutil.copyfile(final_ckpt, default_ckpt)
            print(f"Copied {final_ckpt} -> {default_ckpt}")
        except OSError as e:
            print(f"Warning: could not copy final checkpoint to {default_ckpt}: {e}")
    
