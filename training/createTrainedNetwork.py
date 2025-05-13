import torch
from training.CustomDataset import CustomDataset,  collate_fn_new_fixed_content_in_style, collate_fn_same_timing
from training.network.Network import Network
from training.network.DiffuseNetwork import FacialDiffusionModel

import random
from process_dataset.Constant import *
from torch.utils.data import DataLoader
import gc


def main():
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print("device:", DEVICE)

    from torch.utils.tensorboard import SummaryWriter
    # path_emotionEncoder = "/home/jbok6825/FacialMotionSynthesisProject/Model_emotionEncoder_normalized_clipping_range5_emotion8/model_199epoch.pth"
    EPOCH_NUM = 300

    # Expriment 1: style content 둘다 랜덤길이 랜덤시점, 근데 style이 content보다 훨씬 더 긴 범위에서 랜덤하게

    batch_size=64


    dataset = CustomDataset(device = DEVICE, path_dataset_dir=PATH_DB+"_clipping_random_big")
    # dataloader_1 = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn_new_fixed_content_in_style, drop_last=True)
    # dataloader_2 = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn_new_fixed_content_in_style, drop_last=True)
    dataloader_1 = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn_same_timing, drop_last=True)
    dataloader_2 = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn_same_timing, drop_last=True)
    


    writer = SummaryWriter()
    dir_model_save = "/home/jbok6825/FacialMotionSynthesisProject/Model/Model_Ours_NoCross"
    network = create_expanded_network(style_latent_dim=12*16, content_vae_mode = False).to(DEVICE)
    
    network.train_network(dataloader=dataloader_1,
                          dataloader_sub = dataloader_2,
                            epoch_num=EPOCH_NUM,
                            writer=writer,
                            save_dir=dir_model_save,
                            save_step=50,
                            mode="NoCross",
                            )
    
    del network
    del writer
    del dataset
    del dataloader_1
    del dataloader_2
    gc.collect()
    torch.cuda.empty_cache()

    # dataset = CustomDataset(device = DEVICE, path_dataset_dir=PATH_DB+"_clipping_random")
    # # dataset = CustomDataset(device = DEVICE, path_dataset_dir=PATH_DB+"_notclipping")
    # dataloader_1 = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn_new_fixed_content_in_style, drop_last=True)
    # dataloader_2 = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn_new_fixed_content_in_style, drop_last=True)


    # writer = SummaryWriter()
    # dir_model_save = "/home/jbok6825/FacialMotionSynthesisProject/Model_Ours_NoAlign_positionalEncoding_random"
    # network = create_expanded_network(use_vae=False, use_normalize=True, positional_encoding=True).to(DEVICE)
    
    # network.train_network(dataloader=dataloader_1,
    #                       dataloader_sub = dataloader_2,
    #                     epoch_num=EPOCH_NUM,
    #                     writer=writer,
    #                     save_dir=dir_model_save,
    #                     save_step=30,
    #                     mode="NoAlign",
    #                     )
    

    # del network
    # del writer
    # del dataset
    # del dataloader_1
    # del dataloader_2
    # gc.collect()
    # torch.cuda.empty_cache()

    # dataset = CustomDataset(device = DEVICE, path_dataset_dir=PATH_DB+"_clipping_random")
    # # dataset = CustomDataset(device = DEVICE, path_dataset_dir=PATH_DB+"_notclipping")
    # dataloader_1 = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn_new_fixed_content_in_style, drop_last=True)
    # dataloader_2 = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn_new_fixed_content_in_style, drop_last=True)


    # writer = SummaryWriter()
    # dir_model_save = "/home/jbok6825/FacialMotionSynthesisProject/Model_Ours_NoConsistency_positionalEncoding_random"
    # network = create_expanded_network(use_vae=False, use_normalize=True, positional_encoding=True).to(DEVICE)
    
    # network.train_network(dataloader=dataloader_1,
    #                       dataloader_sub = dataloader_2,
    #                     epoch_num=EPOCH_NUM,
    #                     writer=writer,
    #                     save_dir=dir_model_save,
    #                     save_step=30,
    #                     mode="NoConsistnecy",
    #                     )
    
    # del network
    # del writer
    # del dataset
    # del dataloader_1
    # del dataloader_2
    # gc.collect()
    # torch.cuda.empty_cache()

    # dataset = CustomDataset(device = DEVICE, path_dataset_dir=PATH_DB+"_clipping_random")
    # # dataset = CustomDataset(device = DEVICE, path_dataset_dir=PATH_DB+"_notclipping")
    # dataloader_1 = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn_new_fixed_content_in_style, drop_last=True)
    # dataloader_2 = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn_new_fixed_content_in_style, drop_last=True)


    # writer = SummaryWriter()
    # dir_model_save = "/home/jbok6825/FacialMotionSynthesisProject/Model_Ours_NoCross_positionalEncoding_random"
    # network = create_expanded_network(use_vae=False, use_normalize=True, positional_encoding=True).to(DEVICE)
    
    # network.train_network(dataloader=dataloader_1,
    #                       dataloader_sub = dataloader_2,
    #                     epoch_num=EPOCH_NUM,
    #                     writer=writer,
    #                     save_dir=dir_model_save,
    #                     save_step=30,
    #                     mode="NoCross",
    #                     )





    # dataset_smpl = CustomDataset(device = DEVICE, path_dataset_dir=PATH_DB+"_only_smpl_random", smpl_mode=True)
    # dataloader_1 = DataLoader(dataset_smpl, batch_size, shuffle=True, collate_fn=collate_fn_new_fixed_content_in_style, drop_last=True)
    # dataloader_2 = DataLoader(dataset_smpl, batch_size, shuffle=True, collate_fn=collate_fn_new_fixed_content_in_style, drop_last=True)
    
    # writer = SummaryWriter()
    # dir_model_save = "/home/jbok6825/FacialMotionSynthesisProject/Model_Ours_SMPL_positionalEncoding_random"
    # network_smpl = create_smpl_version_network()

    # network_smpl.train_network(dataloader=dataloader_1,
    #                       dataloader_sub = dataloader_2,
    #                     epoch_num=EPOCH_NUM,
    #                     writer=writer,
    #                     save_dir=dir_model_save,
    #                     save_step=30,
    #                     mode="origin"
    #                     )
    
    # del network_smpl
    # del writer
    # del dataset_smpl
    # del dataloader_1
    # del dataloader_2
    # gc.collect()
    # torch.cuda.empty_cache()

    # dataset_same = CustomDataset(device = DEVICE, path_dataset_dir=PATH_DB+"_only_same_random", same_mode=True)
    # dataloader_1 = DataLoader(dataset_same, batch_size, shuffle=True, collate_fn=collate_fn_new_fixed_content_in_style, drop_last=True)
    # dataloader_2 = DataLoader(dataset_same, batch_size, shuffle=True, collate_fn=collate_fn_new_fixed_content_in_style, drop_last=True)
    
    # writer = SummaryWriter()
    # dir_model_save = "/home/jbok6825/FacialMotionSynthesisProject/Model_Ours_SAME_positionalEncoding_random"
    # network_same = create_same_version_network()

    # network_same.train_network(dataloader=dataloader_1,
    #                       dataloader_sub = dataloader_2,
    #                     epoch_num=EPOCH_NUM,
    #                     writer=writer,
    #                     save_dir=dir_model_save,
    #                     save_step=30,
    #                     mode="origin"
    #                     )
    
    # del network_same
    # del writer
    # del dataset_same
    # del dataloader_1
    # del dataloader_2
    # gc.collect()
    # torch.cuda.empty_cache()

    

    # writer = SummaryWriter("./runs/Model_NoCrossLoss")
    # dir_model_save = "/home/jbok6825/FacialMotionSynthesisProject/Model_Ours_NoCrossLoss"
    # network = create_expanded_network().to(DEVICE)
    
    # network.train_network(dataloader=dataloader_1,
    #                       dataloader_sub = dataloader_2,
    #                     epoch_num=EPOCH_NUM,
    #                     writer=writer,
    #                     save_dir=dir_model_save,
    #                     save_step=30,
    #                     mode="NoCross"
    #                     )
    
    # del network
    # del writer
    # gc.collect()
    # torch.cuda.empty_cache()
    

    # writer = SummaryWriter("./runs/Model_NoConLoss")
    # dir_model_save = "/home/jbok6825/FacialMotionSynthesisProject/Model_Ours_NoConLoss"
    # network = create_expanded_network().to(DEVICE)
    
    # network.train_network(dataloader=dataloader_1,
    #                       dataloader_sub = dataloader_2,
    #                     epoch_num=EPOCH_NUM,
    #                     writer=writer,
    #                     save_dir=dir_model_save,
    #                     save_step=30,
    #                     mode="NoConsistnecy"
    #                     )
    

    
  

    ## Expriment 4: content는 fixed size 랜덤시점, style은 그것보다 긴 범위에서 랜덤하게 

    # del dataloader_1
    # del dataloader_2
    # del dataset_smpl
    # del network
    # del writer

    # dataset = CustomDataset(device = DEVICE, path_dataset_dir=PATH_DB+"_clipping_range5_emotion8")

    # dataloader_1 = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn_new_fixed_content_in_style, drop_last=True)
    # dataloader_2 = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn_new_fixed_content_in_style, drop_last=True)
    
    # writer = SummaryWriter("./runs/Model_Ours_no_consistency")

    
    # dir_model_save = "/home/jbok6825/FacialMotionSynthesisProject/Model_Ours"
    # network = create_expanded_network().to(DEVICE)

    # network.train_network(dataloader=dataloader_1,
    #                       dataloader_sub = dataloader_2,
    #                     epoch_num=EPOCH_NUM,
    #                     writer=writer,
    #                     save_dir=dir_model_save,
    #                     save_step=30,
    #                     emotionEncoder_fixed=False,
    #                     mode_velocity=False,
    #                     mode_disentanglement=False,
    #                     reconstruction_jaw_weight = 1000,
    #                     )
    




def create_diffuse_network(path_preTrainedModel = None):
    size_fullbody_motion_feature = 144
    size_facial_motion_feature = 53
    emotion_latent_dim = 32
    face_content_latent_dim = 512
    body_content_latnet = 512


    network = FacialDiffusionModel( 
                      face_dim = size_facial_motion_feature, 
                      body_dim = size_fullbody_motion_feature, 
                      face_style_latent_dim=emotion_latent_dim,
                      content_latent_dim=body_content_latnet,
                      ).to(DEVICE)
    


    if path_preTrainedModel != None:
        pretrained_model = torch.load(path_preTrainedModel)
        network.load_state_dict(pretrained_model.state_dict())
        # network.eval()

    return network
    



        
     
def create_expanded_network(path_preTrainedModel = None, use_normalize = True, positional_encoding = True, style_latent_dim = 128, content_vae_mode = False):
    size_fullbody_motion_feature = 144
    size_facial_motion_feature = 53
    emotion_latent_dim = style_latent_dim
    face_content_latent_dim = 512
    body_content_latnet = 512

    network = Network(DEVICE, 
                      face_dim = size_facial_motion_feature, 
                      body_dim = size_fullbody_motion_feature, 
                      face_emotion_latent_dim=style_latent_dim,
                      face_content_latent_dim=face_content_latent_dim,
                      body_content_latent_dim=body_content_latnet,
                      use_normalize = use_normalize,
                      vae_mode= True,
                      positional_encoding=positional_encoding,
                      content_vae_mode=content_vae_mode
                      ).to(DEVICE)
    
    if path_preTrainedModel != None:
        pretrained_model = torch.load(path_preTrainedModel)
        network.load_state_dict(pretrained_model.state_dict())
        # network.eval()

    return network

def create_same_version_network(path_preTrainedModel = None, body_content_latent_dim = 32):
    size_fullbody_motion_feature = 32
    size_facial_motion_feature = 53
    emotion_latent_dim = 32
    face_content_latent_dim = body_content_latent_dim
    body_content_latent_dim = body_content_latent_dim

    network = Network(DEVICE, 
                      face_dim = size_facial_motion_feature, 
                      body_dim = size_fullbody_motion_feature,
                      face_emotion_latent_dim=emotion_latent_dim,
                      face_content_latent_dim=face_content_latent_dim,
                      body_content_latent_dim=body_content_latent_dim,
                      same_mode = True,

                      ).to(DEVICE)
    
    if path_preTrainedModel != None:
        pretrained_model = torch.load(path_preTrainedModel)
        network.load_state_dict(pretrained_model.state_dict())
        network.eval()

    return network

def create_smpl_version_network(path_preTrainedModel = None, path_emotionEncoder = None):
    size_fullbody_motion_feature = len(DATASET_EXTRACT_JOINT_LIST)*3
    size_facial_motion_feature = 53
    emotion_latent_dim = 32
    face_content_latent_dim = 512
    body_content_latnet = 512

    network = Network(DEVICE, 
                      face_dim = size_facial_motion_feature, 
                      body_dim = size_fullbody_motion_feature, 
                      face_emotion_latent_dim=emotion_latent_dim,
                      face_content_latent_dim=face_content_latent_dim,
                      body_content_latent_dim=body_content_latnet,
                      path_emotionEncoder=path_emotionEncoder,
                      ).to(DEVICE)
    
    if path_preTrainedModel != None:
        pretrained_model = torch.load(path_preTrainedModel)
        network.load_state_dict(pretrained_model.state_dict())
        network.eval()

    return network
