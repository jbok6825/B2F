import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# Z축이 위로 향하도록 하는 회전 행렬 (90도 회전)
def adjust_orientation(data, sampled_indices):
    rotation_matrix = R.from_euler('x', -90, degrees=True).as_matrix()

    # root_orient 변환
    root_orient = data['root_orient'][sampled_indices.astype(int), :3]
    adjusted_orient = R.from_rotvec(root_orient).as_matrix()
    adjusted_orient = np.einsum('ij,bjk->bik', rotation_matrix, adjusted_orient)  # Apply rotation
    adjusted_orient = R.from_matrix(adjusted_orient).as_rotvec()

    # trans 변환
    trans = data['trans'][sampled_indices.astype(int), :3]
    adjusted_trans = (rotation_matrix @ trans.T).T  # Apply rotation to translation

    return adjusted_orient, adjusted_trans

def adjust_fps(data, current_fps, target_fps):
    # 샘플링 비율 계산
    step = int(current_fps / target_fps)
    sampled_indices = np.arange(0, data['root_orient'].shape[0], step)
    return sampled_indices

test_motion_list = [
    "/home/jbok6825/다운로드/CMU/05/05_07_stageii.npy",
    
    
    ]

for i in range(len(test_motion_list)):

    # npz 파일 경로


    input_file = test_motion_list[i] # .npz 파일 경로
    output_file = '/home/jbok6825/dataset_test/cmu/'+ "test_"+str(i+49)+".npy"  # 저장될 .npy 파일 경로
    current_fps = 90
    target_fps = 30

    # .npz 파일 로드
    data = np.load(input_file, allow_pickle=True)

    # 프레임 개수 설정

    sampled_indices = adjust_fps(data, current_fps, target_fps)
    print(len(sampled_indices))

    # root_orient과 trans 변환
    adjusted_orient, adjusted_trans = adjust_orientation(data, sampled_indices)

    # motion 배열 생성
    frame_count = len(sampled_indices)
    motion = np.zeros((frame_count, 309 + 3))  # 전체 feature 크기

    motion[:, :3] = adjusted_orient  # 변환된 root_orient
    motion[:, 3:66] = data['pose_body'][sampled_indices, :63]
    motion[:, 66:156] = data['pose_hand'][sampled_indices, :90]
    motion[:, 156:159] = data['pose_jaw'][sampled_indices, :3]
    motion[:, 159:209] = np.zeros((frame_count, 50))  # face_expr 초기화
    motion[:, 309:312] = adjusted_trans + 0.4  # 변환된 trans

    # 변환된 motion 데이터를 저장
    np.save(output_file, motion)

    print(f"{current_fps} FPS 데이터를 {target_fps} FPS로 변환하여 저장했습니다: {output_file}")

