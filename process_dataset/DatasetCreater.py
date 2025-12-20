from process_dataset.FileManager import FileManager
import process_dataset.utils as utils
import numpy as np
from CharacterAnimationTools.anim import amass
from CharacterAnimationTools.anim import bvh
from scipy.spatial.transform import Rotation as R
from CharacterAnimationTools.util import quat
from process_dataset.Constant import *
import torch
import pickle

"""
amass.py
AMASS :
    Format: ***.npz (numpy binary file)
    Parameters:
        "trans" (np.ndarray): Root translations. shape: (num_frames, 3).
        "gender" (np.ndarray): Gender name. "male", "female", "neutral".
        "mocap_framerate" (np.ndarray): Fps of mocap data.
        "betas" (np.ndarray): PCA based body shape parameters. shape: (num_betas=16).
        "dmpls" (np.ndarray): Dynamic body parameters of DMPL. We do not use. Unused for skeleton. shape: (num_frames, num_dmpls=8).
        "poses" (np.ndarray): SMPLH pose parameters (rotations). shape: (num_frames, num_J * 3 = 156).
"""

def get_extract_joint_index_list():

    extract_joint_index_list = []
    
    test_file_path = "/home/jbok6825/dataset_MotionX/dance/subset_0000/A_Han_And_Tang_Dance_That_You_Will_Never_Get_Tired_Of.npy"
    motion = np.load(test_file_path)
    smplh = utils.motionX2smplh(motion)
    anim = amass.load(
                amass_motion_file=smplh,
                remove_betas=True,
                gender="neutral",
                anim_name=test_file_path.split("/")[-1],
                load_hand = True
            )
    # anim2 = bvh.load(filepath = "/home/jbok6825/physionet.org/files/kinematic-actors-emotions/2.1.0/BVH/F01/F01A0V1.bvh",
    #                  )

    # print(len(anim.joint_names))
    # print()
    # print(len(anim2.joint_names))

    # # bvh.save(
    # #                 filepath= "test" + ".bvh",
    # #                 anim=anim2
    # #             )
    # exit()
    
    for joint in DATASET_EXTRACT_JOINT_LIST:
        extract_joint_index_list.append(anim.joint_names.index(joint))

    return extract_joint_index_list


def main():
    # anim = bvh.load(filepath="test.bvh")
    fileManager = FileManager()
    extract_joint_index_list = get_extract_joint_index_list()


    face_expr_feature_list = torch.tensor([], device = DEVICE)
    jaw_feature_list = torch.tensor([], device = DEVICE)
    style_code_list = []
    local_position_feature_list = torch.tensor([], device = DEVICE)
    local_orientation_feature_list = torch.tensor([], device = DEVICE)
    local_velocity_feature_list = torch.tensor([], device = DEVICE)
    MIN_LENGTH = 180  # 8초
    OVERLAP_LENGTH = 60 # 2초

    file_num = 0
    dataset_size = 0
    style_code_count = []
    test_code_count = []
    for i in range(0, NUM_STYLE_CODE):
        style_code_count.append(0)
        test_code_count.append(0)
        
    total_length = 0
    test_dataset = []
    training_dataset = []

    import random

    indices = list(range(len(fileManager.file_path_list)))
    random.shuffle(indices)  # file_index 순서 섞기

    for file_index in indices:
        fileManager.file_path_list[file_index]
        file_path = fileManager.file_path_list[file_index]
        style_file_path = fileManager.style_file_path_list[file_index]
    
        motion = np.load(file_path)
        motion_parms = utils.parameterize_motionX(motion)


        if utils.get_frame_legnth(motion_parms) > 180: # frame_length check
            if np.any(np.array(motion_parms["trans"][:, 1]) <= 0) == False and np.any(np.array(motion_parms["trans"][:, 1]) > 1.7) == False: # root_trans check
                if np.max(np.array(motion_parms["face_expr"])) >=1 or np.min(np.array(motion_parms["face_expr"])) <=-1:
                    with open(style_file_path, "r") as f:
                        style = f.readline().strip().lower()
                        style_code = DICT_STYLE_CODE.get(style, -1)
                        
                    if style_code < 0: # style code check
                        continue

                    elif style_code_count[style_code] >= 5000:
                        if test_code_count[style_code] < 500:

                            test_dataset.append(file_path)
                            test_code_count[style_code] = test_code_count[style_code] + 1
                            print("###", test_code_count)
                        continue

                    
                    training_dataset.append(file_path)

                    smplh = utils.motionX2smplh(motion)
                    anim = amass.load(
                        amass_motion_file=smplh,
                        remove_betas=True,
                        gender="neutral",
                        anim_name=file_path.split("/")[-1],
                        load_hand = True
                    )

                    # print(len(anim.skel.joints))
                    # print(len(anim.skel.bone_lengths)) # parent와 자기 사이의 거리 
                    # print(anim.skel.rest_forward)
                    # print(anim.skel.rest_vertical)
                    # print(anim.skel.)
                    # exit()


                    frame_length = utils.get_frame_legnth(motion_parms)
                    # print("frame_length:", frame_length)
                    total_length += frame_length
                    # print("total_length: ", total_length)
                    # print("total time", total_length *(1/30))

                    proj_root_pos =  torch.tensor(anim.proj_root_pos(), device = DEVICE, dtype = torch.float32)/100
                    proj_root_rot = torch.tensor(quat.to_xform(anim.proj_root_rot), device = DEVICE, dtype = torch.float32)
                    character_local_coordinate = utils.get_character_local_coordinate_from_projeted_info(proj_root_pos, proj_root_rot)

                    global_position = torch.tensor(anim.gpos, device = DEVICE)[:, extract_joint_index_list, :]/100 # shape (N, J, 3)
                    global_orientation = torch.tensor(quat.to_xform(anim.grot)[:, extract_joint_index_list, :], device = DEVICE)
                    global_velocity = torch.tensor(anim.gposvel[:, extract_joint_index_list,: ], device = DEVICE)/100

                    current_character_local_position, current_character_local_orientation, current_character_local_velocity= utils.get_motion_feature(global_position, global_orientation, global_velocity, character_local_coordinate, only_current=True)
                    facial_feature = utils.get_facial_feature(motion_parms)
                    current_face_expr_feature_list = facial_feature['face_expr'].to(DEVICE)
                    current_jaw_feature_list = facial_feature['jaw'].to(DEVICE)


                    start_index = 0

                    while start_index + MIN_LENGTH <= frame_length:
                        local_position_feature_list = torch.cat((local_position_feature_list, current_character_local_position[start_index: start_index+MIN_LENGTH].unsqueeze(0)), dim = 0)
                        local_orientation_feature_list = torch.cat((local_orientation_feature_list, current_character_local_orientation[start_index: start_index+MIN_LENGTH].unsqueeze(0)), dim = 0)
                        local_velocity_feature_list = torch.cat((local_velocity_feature_list, current_character_local_velocity[start_index: start_index+MIN_LENGTH].unsqueeze(0)), dim = 0)
                        face_expr_feature_list = torch.cat((face_expr_feature_list, current_face_expr_feature_list[start_index: start_index+MIN_LENGTH].unsqueeze(0)), dim = 0)
                        jaw_feature_list = torch.cat((jaw_feature_list, current_jaw_feature_list[start_index: start_index+MIN_LENGTH].unsqueeze(0)), dim = 0)
                        style_code_list.append(style_code)


                        start_index += OVERLAP_LENGTH
                        dataset_size = dataset_size + 1
                        style_code_count[style_code] = style_code_count[style_code] + 1

                        print("!", style_code_count)

                        



                print("##", local_position_feature_list.shape[0])

                if local_position_feature_list.shape[0] > 2000:

                    print("local position feature list shape", local_position_feature_list.shape)
                    print("local orientation feature list shape", local_orientation_feature_list.shape)
                    print("local_velocity_feature_list shape", local_velocity_feature_list.shape)
                    print("face expr shape", face_expr_feature_list.shape)
                    print("jaw shape", jaw_feature_list.shape)
                    print("style code list shape", len(style_code_list))

                    total_data = {
                        'position': local_position_feature_list.cpu(),
                        'orientation': local_orientation_feature_list.cpu(),
                        'velocity': local_velocity_feature_list.cpu(),
                        'face_expr': face_expr_feature_list.cpu(),
                        'jaw': jaw_feature_list.cpu(),
                        'style_code_list': torch.tensor(style_code_list).cpu(),
                    }

                    with open(PATH_DB + "_clipping_random_big/"+"data_"+str(file_num)+".pkl", 'wb') as file:
                        pickle.dump(total_data, file)

                    del local_position_feature_list
                    del local_orientation_feature_list
                    del local_velocity_feature_list
                    del face_expr_feature_list
                    del jaw_feature_list
                    del style_code_list

                    torch.cuda.empty_cache()

                    local_position_feature_list = torch.tensor([], device = DEVICE)
                    local_orientation_feature_list = torch.tensor([], device = DEVICE)
                    local_velocity_feature_list = torch.tensor([], device = DEVICE)
                    face_expr_feature_list = torch.tensor([], device = DEVICE)
                    jaw_feature_list = torch.tensor([], device = DEVICE)
                    style_code_list = []

                    file_num += 1

    if local_position_feature_list.shape[0] > 0:

        print("local position feature list shape", local_position_feature_list.shape)
        print("local orientation feature list shape", local_orientation_feature_list.shape)
        print("local_velocity_feature_list shape", local_velocity_feature_list.shape)
        print("face expr shape", face_expr_feature_list.shape)
        print("jaw shape", jaw_feature_list.shape)
        print("style code list shape", len(style_code_list))

        total_data = {
            'position': local_position_feature_list.cpu(),
            'orientation': local_orientation_feature_list.cpu(),
            'velocity': local_velocity_feature_list.cpu(),
            'face_expr': face_expr_feature_list.cpu(),
            'jaw': jaw_feature_list.cpu(),
            'style_code_list': torch.tensor(style_code_list).cpu(),
        }

        with open(PATH_DB + "_clipping_random_big/"+"data_"+str(file_num)+".pkl", 'wb') as file:
            pickle.dump(total_data, file)


    with open('/home/jbok6825/FacialMotionSynthesisProject/test_dataset_list_clipping_random_big.txt', 'w') as file:
        for path in test_dataset:
            file.write(path + '\n')

    with open('/home/jbok6825/FacialMotionSynthesisProject/training_dataset_list_big.txt', 'w') as file:
        for path in training_dataset:
            file.write(path + '\n')
            