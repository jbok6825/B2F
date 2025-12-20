import os
import numpy as np
from CharacterAnimationTools.anim import amass
from CharacterAnimationTools.anim import bvh
import process_dataset.utils as utils




common_npy_path = "/home/jbok6825/dataset_test/cmu/npy"
common_bvh_path = "/home/jbok6825/dataset_test/cmu/bvh"

file_num = 49

for index in range(file_num):
    index = 34
    test_file_name = "test_" + str(index)
    index = 34
    
    path_test_smpl_file = common_npy_path + "/" + test_file_name+".npy"
    path_test_bvh_file = common_bvh_path + "/" + test_file_name + ".bvh"
    motion = np.load(path_test_smpl_file)

    smplh = utils.motionX2smplh(motion)
    anim = amass.load(
        amass_motion_file=smplh,
        remove_betas=True,
        gender="neutral",
        anim_name=path_test_smpl_file.split("/")[-1],
        load_hand = False
    )

    bvh.save(
        filepath=path_test_bvh_file,
        anim=anim
    )
    print(path_test_bvh_file)
    exit()

