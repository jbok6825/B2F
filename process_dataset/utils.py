import torch
from process_dataset.Constant import * 
import numpy as np
from scipy.spatial.transform import Rotation as R

def parameterize_motionX(motion, max_size = None):

    max_size = None
    if max_size == None or motion.shape[0] < max_size:
        max_size = motion.shape[0]
    motion_parms = {
            'root_orient': torch.tensor(motion[:max_size, :3]),  # controls the global root orientation
            'pose_body': torch.tensor(motion[:max_size, 3:3+63]),  # controls the body
            'pose_hand': torch.tensor(motion[:max_size, 66:66+90]),  # controls the finger articulation
            'pose_jaw': torch.tensor(motion[:max_size, 66+90:66+93]),  # controls the yaw pose
            # 'face_expr': torch.tensor(np.clip(motion[:, 159:159+50], -2, 2)),
            'face_expr': torch.tensor(np.clip(motion[:max_size, 159:159+50], -5, 5)), # controls the face expression
            'trans': torch.tensor(motion[:max_size, 309:309+3]),  # controls the global body position
            # 'poses': torch.zeros((motion.shape[0], 165)),
            'poses': torch.zeros((max_size), 165),
            # 'face_shape': torch.zeros((motion.shape[0]), 100),
            'face_shape': torch.zeros((max_size), 100),
            'betas': torch.zeros(10, ),
            'gender': "neutral",
            'mocap_frame_rate': 30,
            'SMPL_version':"SMPLX"
    }
    motion_parms['poses'][:, :3] = motion_parms['root_orient']
    motion_parms['poses'][:, 3:3+63] = motion_parms['pose_body']
    motion_parms['poses'][:, 66:66+3] = motion_parms['pose_jaw']
    motion_parms['poses'][:, 75:75+90] = motion_parms['pose_hand']




    return motion_parms

def get_facial_feature(motion_parms, start_frame= None, end_frame = None):

    if start_frame == None and end_frame == None:
        facial_feature = {
            'face_expr': torch.tensor(motion_parms['face_expr']),
            'jaw': torch.tensor(motion_parms['pose_jaw'])
        }
    else:
        facial_feature = {
            'face_expr': torch.tensor(motion_parms['face_expr'][start_frame:end_frame]),
            'jaw': torch.tensor(motion_parms['pose_jaw'][start_frame:end_frame])
        }

    return facial_feature

def get_frame_legnth(motion_parms):
    frame_length = motion_parms['pose_body'].shape[0]
    return frame_length


def motionX2smplh(motion):
    smplh = {
            'root_orient': np.array(motion[:, :3]),  # controls the global root orientation
            'pose_body': np.array(motion[:, 3:3+63]),  # controls the body
            'pose_hand': np.array(motion[:, 66:66+90]),  # controls the finger articulation
            'pose_jaw': np.array(motion[:, 66+90:66+93]),  # controls the yaw pose
            'face_expr': np.array(motion[:, 159:159+10]),  # controls the face expression
            'trans': np.array(motion[:, 309:309+3]),  # controls the global body position
            'poses': np.array(torch.zeros((motion.shape[0], 156))),
            'betas': np.array(torch.zeros((16))),
            'gender': "female",
            'mocap_framerate': 30
    }
    smplh['poses'][:, :3] = smplh['root_orient']
    smplh['poses'][:, 3:3+63] = smplh['pose_body']
    smplh['poses'][:, 66:66+90] = smplh['pose_hand']

    return smplh


def l2norm_tensor(v):
    return torch.norm(v, dim = 1)

def normalized_tensor(v):
    return torch.nn.functional.normalize(v)

def get_local_coordinate_point_value(local_coordinate_list, global_coordinate_point_value_list):

    global_value_list = torch.ones([global_coordinate_point_value_list.shape[0], 4], dtype = torch.float32, device = DEVICE)
    global_value_list[:, :3] = global_coordinate_point_value_list[:, :3]
    global_value_list = global_value_list.unsqueeze(2)

    local_value_list = (torch.linalg.inv(local_coordinate_list) @ global_value_list).view(-1, 4)

    return local_value_list[:, :3]

def get_local_coordinate_vector_value(local_coordinate_list, global_coordinate_vector_value_list):
    global_value_list = torch.zeros([global_coordinate_vector_value_list.shape[0], 4], dtype=torch.float32, device = DEVICE)
    global_value_list[:, :3] = global_coordinate_vector_value_list[:, :3]
    global_value_list = global_value_list.unsqueeze(2)
    local_value_list = (torch.linalg.inv(local_coordinate_list) @ global_value_list).view(-1, 4)

    return local_value_list[:, :3]

def get_local_orientation_value(local_coordinate_list, global_orientation_list):
    return torch.linalg.inv(local_coordinate_list)[:, :3, :3] @ global_orientation_list

def get_global_orientation_value(basis_coordinate_list, local_orientation_list):
    return basis_coordinate_list[:, :3, :3] @ local_orientation_list

def get_global_point_value(basis_coordinate_list, local_coordinate_point_value_list):

    local_value_list = torch.ones([local_coordinate_point_value_list.shape[0], 4], dtype = torch.float32, device=DEVICE)
    local_value_list[:, :3] = local_coordinate_point_value_list[:, :3]
    local_value_list = local_value_list.unsqueeze(2)

    global_value_list = (basis_coordinate_list @ local_value_list).view(-1, 4)

    return global_value_list[:, :3]

def get_global_coordinate_vector_value(local_coordinate_list, local_coordinate_vector_value_list):
    local_value = torch.zeros([local_coordinate_vector_value_list.shape[0], 4], dtype = torch.float32, device =DEVICE)
    local_value[:, :3] = local_coordinate_vector_value_list[:, :3]
    local_value = local_value.unsqueeze(2)

    global_value = (local_coordinate_list @ local_value).view(-1, 4)

    return global_value[:, :3]

def blender_matrix_to_opengl_matrix(mb):
    mb = np.array(mb)
    mo = np.array([
        [mb[0, 0], mb[0, 2], -1 * mb[0, 1]],
        [mb[2, 0], mb[2, 2], -1 * mb[2, 1]],
        [-1 * mb[1, 0], -1 * mb[1, 2], mb[1, 1]] 
    ])

    return mo

def blender_matrix_to_opengl_rotvec(mb):
    mb = np.array(mb)
    mo = np.array([
        [mb[0, 0], mb[0, 2], -1 * mb[0, 1]],
        [mb[2, 0], mb[2, 2], -1 * mb[2, 1]],
        [-1 * mb[1, 0], -1 * mb[1, 2], mb[1, 1]] 
    ])
    r = R.from_matrix(mo)
    
    # Get the rotation vector (axis-angle representation)
    rotation_vector = r.as_rotvec()
    
    return rotation_vector

    return mo

def blender_position_to_opengl(pb):
    pb = np.array(pb)
    po = np.array([pb[0], pb[2], -1 * pb[1]])

    return po

def get_character_local_coordinate(global_root_position_list, global_root_orientation_list):

    num_posture = len(global_root_orientation_list)
    newOrigin = global_root_position_list.clone().detach()
    newOrigin[:, 1] = 0.

    direction_list = get_character_global_direction(global_root_orientation_list)
    
    newZaxis = direction_list # shape (N, 3)
    newYaxis = torch.zeros((num_posture, 3), dtype = torch.float32, device = DEVICE)
    newYaxis[:, 1] = 1. # shape (N, 3)
    newXaxis = normalized_tensor(torch.linalg.cross(newYaxis, newZaxis))

    character_local_coordinate = torch.zeros((num_posture, 4, 4), dtype=torch.float32, device = DEVICE)
    character_local_coordinate[:, 3, 3] = 1
    character_local_coordinate[:, :3, 0] = newXaxis
    character_local_coordinate[:, :3, 1] = newYaxis
    character_local_coordinate[:, :3, 2] = newZaxis
    character_local_coordinate[:, :3, 3] = newOrigin

    return character_local_coordinate


def get_character_global_direction(global_root_orientation_list):

    newDirection_list = global_root_orientation_list[:, :3, 2].clone().detach()
    newDirection_list[:, 1] = 0.
    newDirection_list = normalized_tensor(newDirection_list)

    return newDirection_list

def get_character_local_coordinate_from_projeted_info(global_projected_root_position_list, global_projected_root_orientation_list):

    num_posture = len(global_projected_root_orientation_list)

    character_local_coordinate = torch.zeros((num_posture, 4, 4), device=DEVICE) # shape (N-10, 4, 4)
    character_local_coordinate[:, 3, 3] = 1.
    character_local_coordinate[:, :3, :3] = global_projected_root_orientation_list
    character_local_coordinate[:, :3, 3] = global_projected_root_position_list
    

    return character_local_coordinate

def get_6d_rotation_from_rotation_matrix_tensor(rotation_matrix_tensor):
    rotation_matrix_tensor = rotation_matrix_tensor[:,:,:2].clone() # shape (N, 3, 3) -> (N, 3, 2)
    rotation_matrix_tensor = rotation_matrix_tensor.transpose(1, 2)

    rotation_matrix_tensor = rotation_matrix_tensor.reshape((-1, 6))

    return rotation_matrix_tensor

def rotation_vector_to_rotation_matrix(rotation_vector_tensor):
    # Rotation vector의 크기 (N, 3)
    theta = torch.norm(rotation_vector_tensor, dim=-1, keepdim=True)  # (N, 1)
    theta = torch.clamp(theta, min=1e-6)  # 작은 값은 안정성을 위해 clamp

    normalized_vector = rotation_vector_tensor / theta  # 방향 벡터 (N, 3)
    cos_theta = torch.cos(theta).unsqueeze(-1)  # (N, 1, 1)로 확장
    sin_theta = torch.sin(theta).unsqueeze(-1)  # (N, 1, 1)로 확장
    
    # skew-symmetric matrix 계산
    K = torch.zeros(rotation_vector_tensor.size(0), 3, 3).to(rotation_vector_tensor.device)  # (N, 3, 3)
    K[:, 0, 1] = -normalized_vector[:, 2]
    K[:, 0, 2] = normalized_vector[:, 1]
    K[:, 1, 0] = normalized_vector[:, 2]
    K[:, 1, 2] = -normalized_vector[:, 0]
    K[:, 2, 0] = -normalized_vector[:, 1]
    K[:, 2, 1] = normalized_vector[:, 0]

    # Rotation matrix 계산 (Rodrigues' rotation formula)
    identity = torch.eye(3).to(rotation_vector_tensor.device).unsqueeze(0)  # (1, 3, 3)
    identity = identity.repeat(rotation_vector_tensor.size(0), 1, 1)  # (N, 3, 3)

    outer_product = torch.bmm(normalized_vector.unsqueeze(2), normalized_vector.unsqueeze(1))  # (N, 3, 1) x (N, 1, 3) = (N, 3, 3)
    
    rotation_matrix = cos_theta * identity + (1 - cos_theta) * outer_product + sin_theta * K  # (N, 3, 3)
    

    return rotation_matrix

# Rotation vector에서 6D rotation representation으로 변환하는 함수
# def get_6d_rotation_from_rotation_vector(rotation_vector_tensor):
#     # Step 1: Rotation vector -> Rotation matrix 변환
#     rotation_matrix_tensor = rotation_vector_to_rotation_matrix(rotation_vector_tensor)
    
#     # Step 2: Rotation matrix에서 6D representation 추출
#     rotation_matrix_tensor = get_6d_rotation_from_rotation_matrix_tensor(rotation_matrix_tensor)

#     return rotation_matrix_tensor


# def get_rotation_matrix_from_6d_rotation(rotation_6d_tensor):
#     # Step 1: (N, 6) -> (N, 3, 2)로 변환
#     rotation_matrix_tensor = rotation_6d_tensor.view(-1, 2, 3).transpose(1, 2)  # (N, 2, 3) -> (N, 3, 2)
    
#     # Step 2: Gram-Schmidt 과정을 사용해 마지막 열 벡터를 계산
#     b1 = rotation_matrix_tensor[:, :, 0]  # 첫 번째 열 (N, 3)
#     b2 = rotation_matrix_tensor[:, :, 1]  # 두 번째 열 (N, 3)
    
#     b1 = b1 / torch.norm(b1, dim=-1, keepdim=True)  # 정규화된 첫 번째 열 (N, 3)
#     b2 = b2 - torch.sum(b1 * b2, dim=-1, keepdim=True) * b1  # b1에 직교하도록 b2 수정
#     b2 = b2 / torch.norm(b2, dim=-1, keepdim=True)  # 정규화된 두 번째 열 (N, 3)
    
#     b3 = torch.cross(b1, b2, dim=-1)  # 마지막 열은 b1, b2의 외적 (N, 3)
    
#     # 회전 행렬 (N, 3, 3)로 결합
#     rotation_matrix = torch.stack([b1, b2, b3], dim=-1)  # (N, 3, 3)
    
#     return rotation_matrix

# 회전 행렬에서 회전 벡터로 변환하는 함수
def rotation_matrix_to_rotation_vector(rotation_matrix_tensor):
    # Step 1: 회전 행렬의 trace (대각합) 계산
    trace = torch.sum(torch.diagonal(rotation_matrix_tensor, dim1=-2, dim2=-1), dim=-1)  # (N)
    
    # Step 2: 회전 각도 계산
    theta = torch.acos(torch.clamp((trace - 1) / 2, min=-1 + 1e-6, max=1 - 1e-6))  # (N)
    
    # Step 3: 회전 벡터 계산
    sin_theta = torch.sin(theta)  # (N)
    theta = theta.unsqueeze(-1)  # (N, 1)
    
    # 각도를 0에 가까운 경우 예외 처리
    near_zero = sin_theta.abs() < 1e-6
    
    rotation_vector = torch.zeros_like(rotation_matrix_tensor[:, :, 0])  # (N, 3)
    
    if not near_zero.all():  # 회전 각도가 충분히 큰 경우
        factor = theta / (2 * sin_theta.unsqueeze(-1))  # (N, 1)
        rotation_vector = factor * torch.stack([
            rotation_matrix_tensor[:, 2, 1] - rotation_matrix_tensor[:, 1, 2],
            rotation_matrix_tensor[:, 0, 2] - rotation_matrix_tensor[:, 2, 0],
            rotation_matrix_tensor[:, 1, 0] - rotation_matrix_tensor[:, 0, 1]
        ], dim=-1)  # (N, 3)
    
    return rotation_vector

# 6D 회전 표현에서 회전 벡터로 변환하는 함수
# def get_rotation_vector_from_6d_rotation(rotation_6d_tensor):
#     # Step 1: 6D -> 3x3 회전 행렬 변환
#     rotation_matrix_tensor = get_rotation_matrix_from_6d_rotation(rotation_6d_tensor)
    
#     # Step 2: 3x3 회전 행렬 -> 회전 벡터 변환
#     rotation_vector_tensor = rotation_matrix_to_rotation_vector(rotation_matrix_tensor)
    
#     return rotation_vector_tensor

def get_orientation_feature(global_orientation, character_local_coordinate):

    num_joint = len(RUNTIME_EXTRACT_JOINT_LIST)
    frame_length = len(global_orientation)
    feature_frame_length = frame_length

    current_global_orientation = global_orientation.clone()
    current_character_local_orientation_matrix = get_local_orientation_value(torch.repeat_interleave(character_local_coordinate, num_joint, dim = 0), current_global_orientation.reshape(-1, 3, 3))
    current_character_local_orientation = get_6d_rotation_from_rotation_matrix_tensor(current_character_local_orientation_matrix).reshape((feature_frame_length, num_joint, 6))

    return current_character_local_orientation

def get_motion_feature(global_position, global_orientation, global_velocity, character_local_coordinate, only_current = True):
    '''
    global_position (N+num_past, J, 3)
    global_orientation (N+num_past, J, 3, 3)
    character_local_coordinate (N, 4, 4)
    '''
    num_joint = len(RUNTIME_EXTRACT_JOINT_LIST)
    frame_length = len(global_position)
    
    if only_current == False:    
        start_index = NUM_PAST_MOTION
    else:
        start_index = 0

    feature_frame_length = frame_length - start_index
    current_global_position = global_position[start_index:]
    current_character_local_position = get_local_coordinate_point_value(torch.repeat_interleave(character_local_coordinate, num_joint, dim = 0), current_global_position.reshape(-1, 3)).reshape(feature_frame_length, num_joint, 3)
    current_global_orientation = global_orientation[start_index:]
    current_character_local_orientation_matrix = get_local_orientation_value(torch.repeat_interleave(character_local_coordinate, num_joint, dim = 0), current_global_orientation.reshape(-1, 3, 3))
    current_character_local_orientation = get_6d_rotation_from_rotation_matrix_tensor(current_character_local_orientation_matrix).reshape((feature_frame_length, num_joint, 6))
    current_global_velocity = global_velocity[start_index:]
    current_character_local_velocity = get_local_coordinate_vector_value(torch.repeat_interleave(character_local_coordinate, num_joint, dim = 0), current_global_velocity.reshape(-1, 3)).reshape(feature_frame_length, num_joint, 3)

    if only_current == False:
        indices = torch.arange(NUM_PAST_MOTION, frame_length).view(-1, 1)
        offsets = torch.arange(-NUM_PAST_MOTION, 0).view(1, -1)
        character_local_coordinate_expand = torch.repeat_interleave(character_local_coordinate, NUM_PAST_MOTION* num_joint, dim = 0) # shape ((N-10)*10*J, 4, 4)()
        past_global_position = global_position[indices + offsets] # shape (N-10, 10, J, 3)
        past_global_position_reshape = past_global_position.reshape(len(character_local_coordinate_expand), 3) 
        past_character_local_position = get_local_coordinate_point_value(character_local_coordinate_expand, past_global_position_reshape).reshape(past_global_position.shape)
        past_global_orientation = global_orientation[indices + offsets]
        past_global_orientation_reshape = past_global_orientation.reshape(len(character_local_coordinate_expand), 3, 3)
        past_character_local_orientation_matrix = get_local_orientation_value(character_local_coordinate_expand, past_global_orientation_reshape)
        past_character_local_orientation = get_6d_rotation_from_rotation_matrix_tensor(past_character_local_orientation_matrix).reshape((feature_frame_length, NUM_PAST_MOTION, num_joint, 6))
        past_global_velocity = global_velocity[indices + offsets]
        past_global_velocity_reshape = past_global_velocity.reshape(len(character_local_coordinate_expand), 3)
        past_character_local_velocity = get_local_coordinate_vector_value(character_local_coordinate_expand, past_global_velocity_reshape).reshape(past_global_velocity.shape)

        return current_character_local_position, current_character_local_orientation, current_character_local_velocity, past_character_local_position, past_character_local_orientation, past_character_local_velocity
    else:
        return current_character_local_position, current_character_local_orientation, current_character_local_velocity

'''
position_feature (frame_length, 3)
orientation fetaure (frame_length, 6)
velocity feature (frame_length, 3)
'''
def get_formatted_data(
        position_feature = None, 
        orientation_feature = None, 
        velocity_feature = None, 
        face_expr_feature = None, 
        jaw_feature = None, 
        face_expr_style_feature = None, 
        jaw_style_feature = None, 
        style_code = None):

    flatten_fullbody_feature = torch.cat((position_feature, orientation_feature, velocity_feature), dim = -1).flatten(1)

    
    if face_expr_feature == None and jaw_feature == None:
        if jaw_style_feature == None and face_expr_style_feature == None:
            return {
                'fullbody_feature': flatten_fullbody_feature.to(DEVICE),
            }
        else:

            flatten_facial_style_feature = torch.cat((jaw_style_feature, face_expr_style_feature), dim = -1)

            return {
                'fullbody_feature': flatten_fullbody_feature.to(DEVICE),
                'facial_style_feature': flatten_facial_style_feature.to(DEVICE),
            }
        
    
    else:
        flatten_facial_style_feature = torch.cat((jaw_style_feature, face_expr_style_feature), dim = -1)
        flatten_facial_feature = torch.cat((jaw_feature, face_expr_feature), dim = -1)
        return {
            'fullbody_feature': flatten_fullbody_feature.to(DEVICE),
            'facial_feature': flatten_facial_feature.to(DEVICE),
            'facial_style_feature': flatten_facial_style_feature.to(DEVICE),
            'style_code': style_code.to(DEVICE)
        }
    

def rotation_vector_to_quaternion(rotation_vector):
    angle = torch.norm(rotation_vector, dim=-1, keepdim=True)
    axis = rotation_vector / (angle + 1e-8)  # Normalize the axis, avoid division by zero
    half_angle = angle * 0.5
    w = torch.cos(half_angle)
    xyz = axis * torch.sin(half_angle)
    quaternion = torch.cat([w, xyz], dim=-1)  # [batch_size, num_frames, 4]
    return quaternion

# 두 쿼터니언 간의 각도 차이(라디안)를 계산하는 함수
def quaternion_angle_difference(q1, q2):
    dot_product = (q1 * q2).sum(dim=-1)
    angle_diff = 2 * torch.acos(torch.clamp(dot_product, -0.99999, 0.99999))
    return angle_diff

# 매 프레임마다의 회전 속도 계산
def compute_rotational_velocity(rotvec_sequence, delta_t=1/30):
    """
    회전 벡터 시퀀스 → angular velocity (방향 + 크기) 시퀀스

    Args:
        rotvec_sequence: np.ndarray, shape (T, 3), axis-angle 회전 벡터
        delta_t: 프레임 간 시간 간격 (초)

    Returns:
        angular_velocity: np.ndarray, shape (T-1, 3), rad/s
    """
    r_seq = R.from_rotvec(rotvec_sequence)  # (T,) Rotation 객체 시퀀스
    r_rel = r_seq[1:] * r_seq[:-1].inv()    # 상대 회전들 (T-1,)

    # 상대 회전을 다시 rotvec으로 → shape (T-1, 3)
    rel_rotvecs = r_rel.as_rotvec()

    # 속도 = 회전벡터 / 시간
    angular_velocity = rel_rotvecs / delta_t  # (T-1, 3)
    return angular_velocity



# def get_mean_l2norm(list1, list2):
#     differences = np.linalg.norm(list1 - list2, axis=2)  # (1412, len(lip_vertex_indices))
#     frame_avg_l2_error = np.mean(differences, axis=1)  # (1412,)

#     # 모든 프레임에 대해 평균 L2 에러 계산
#     overall_avg_l2_error = np.mean(frame_avg_l2_error)

#     return overall_avg_l2_error
def get_mean_l2norm(list1, list2):
    # list1, list2: shape (frame_length, 50)
    differences = np.abs(list1 - list2)  # element-wise 차이, shape: (frame_length, 50)
    frame_avg_l2_error = np.sqrt(np.mean(differences**2, axis=1))  # 프레임별 L2 norm 평균

    overall_avg_l2_error = np.mean(frame_avg_l2_error)  # 전체 프레임 평균
    return overall_avg_l2_error

import numpy as np
from scipy.spatial.transform import Rotation as R

def get_mean_angular_difference(rotvec_pred, rotvec_gt, eps=1e-5):
    """
    상대적인 회전 오차 (percentage)
    - rotvec_pred, rotvec_gt: shape (frame_length, 3)
    - eps: 0 나눗셈 방지를 위한 값

    Returns:
    - 평균 상대 회전 오차 (단위: %)
    """
    pred_rot = R.from_rotvec(rotvec_pred)
    gt_rot = R.from_rotvec(rotvec_gt)
    
    rel_rot = pred_rot * gt_rot.inv()
    
    abs_angle_diff = rel_rot.magnitude()               # radian
    gt_angle = gt_rot.magnitude()
    
    relative_error = abs_angle_diff / (gt_angle + eps)
    return np.mean(relative_error) * 100  # % 단위

# def get_fdd_metric(velocity_list1, velocity_list2):
#     l2_norm_velocity1 = np.linalg.norm(velocity_list1, axis=-1)
#     l2_norm_velocity2 = np.linalg.norm(velocity_list2, axis=-1)

#     dyn_1 = np.std(l2_norm_velocity1 , axis=0)
#     dyn_2 = np.std(l2_norm_velocity2, axis=0)

#     fdd = np.mean(np.abs(dyn_1 - dyn_2))


#     return fdd
def compute_wfdd_per_clip(pred_clip, gt_clip):
    """
    pred_clip, gt_clip: shape (T, D)  # T=frame, D=weight dim
    Returns: scalar FDD-like value
    """
    dyn_pred = np.std(pred_clip, axis=0)  # shape (D,)
    dyn_gt = np.std(gt_clip, axis=0)      # shape (D,)
    return np.mean(np.abs(dyn_pred - dyn_gt))  # scalar

def compute_wfdd_over_all_clips(pred_weights, gt_weights, clip_lengths):
    """
    pred_weights, gt_weights: shape (T_total, D)
    clip_lengths: list of ints, each motion's frame count
    Returns: mean wFDD over all clips
    """
    assert pred_weights.shape == gt_weights.shape
    if sum(clip_lengths) == pred_weights.shape[0]:
        print()
        exit()

    fdd_scores = []
    idx = 0
    for length in clip_lengths:
        pred_clip = pred_weights[idx:idx+length]
        gt_clip = gt_weights[idx:idx+length]
        fdd = compute_wfdd_per_clip(pred_clip, gt_clip)
        fdd_scores.append(fdd)
        idx += length

    return np.mean(fdd_scores)





# def compute_expression_motion_sync_with_direction(vertex_sequence, joint_velocities, clip_lengths, direction_threshold=0.007, expression_threshold=0.0005):
#     # start_idx = 0
#     # sync_scores = []
    
#     # for length in clip_lengths:
#     #     # 현재 클립의 시퀀스 추출
#     #     clip_vertex_sequence = vertex_sequence[start_idx:start_idx + length]
#     #     clip_joint_velocities = joint_velocities[start_idx:start_idx + length - 1]  # 한 프레임 줄이기
        
#     #     # 얼굴 표정의 프레임 간 변위 계산
#     #     expression_displacements = np.linalg.norm(np.diff(clip_vertex_sequence, axis=0), axis=-1).mean(axis=1)  # (clip_length-1,)
#     #     expression_change_points = expression_displacements > expression_threshold  # 표정 급변 프레임 (True/False)


#     #     # 속도 벡터 간의 차이 크기를 사용해 급격한 방향 변화 탐지
#     #     direction_changes = np.linalg.norm(np.diff(clip_joint_velocities, axis=0), axis=-1).mean(axis=1)  # (clip_length-2,)
#     #     velocity_change_points = direction_changes > direction_threshold  # 차이 크기가 임계값을 넘는 경우 급변으로 간주


#     #     # 동기화 프레임 계산
#     #     min_length = min(len(expression_change_points), len(velocity_change_points))  # 최소 길이에 맞추기
#     #     sync_count = np.sum(expression_change_points[:min_length] & velocity_change_points[:min_length])
#     #     total_change_count = np.sum(expression_change_points[:min_length] | velocity_change_points[:min_length])

#     #     # 동시성 비율 계산
#     #     sync_score = sync_count / (total_change_count + 1e-6)
#     #     sync_scores.append(sync_score)
        
#     #     # 다음 클립 시작 인덱스로 이동
#     #     start_idx += length
    
#     # # 전체 시퀀스에 대한 평균 동시성 점수 계산
#     # overall_sync_score = np.mean(sync_scores)
    
#     # return overall_sync_score

#     start_idx = 0
#     consistency_scores = []
#     weights = []

#     for length in clip_lengths:
#         clip_vertex_sequence = vertex_sequence[start_idx:start_idx + length]
#         clip_joint_velocities = joint_velocities[start_idx:start_idx + length - 1]

#         # 얼굴 표정의 변위 변화 (속도) 후 가속도 계산
#         expression_velocity = np.linalg.norm(np.diff(clip_vertex_sequence, axis=0), axis=-1).mean(axis=1)
#         expression_acceleration = np.diff(expression_velocity)

#         # 관절 속도의 가속도 계산
#         joint_accelerations = []
#         for i in range(0, len(clip_joint_velocities) - 1):
#             curr_velocity = clip_joint_velocities[i].flatten()
#             next_velocity = clip_joint_velocities[i + 1].flatten()
#             acceleration = np.linalg.norm(next_velocity - curr_velocity)
#             joint_accelerations.append(acceleration)

#         joint_accelerations = np.array(joint_accelerations)

#         # 얼굴 표정 가속도와 관절 가속도의 상관 계수 계산
#         min_length = min(len(expression_acceleration), len(joint_accelerations))
#         expression_acceleration, joint_accelerations = expression_acceleration[:min_length], joint_accelerations[:min_length]

#         if np.std(expression_acceleration) > 1e-6 and np.std(joint_accelerations) > 1e-6:
#             correlation = np.corrcoef(expression_acceleration, joint_accelerations)[0, 1]
#             consistency_scores.append(correlation)
#             weights.append(length)

#         start_idx += length

#     # 가중 평균 일관성 점수 계산 (length 가중치 적용)
#     overall_consistency_score = np.average(consistency_scores, weights=weights) if weights else 0
    
#     return overall_consistency_score

def compute_expression_motion_sync_with_direction(
    vertex_sequence, joint_velocities, clip_lengths,
    expression_threshold=0.0005, direction_threshold=0.007, target_ratio=1.0):

    """
    몸과 얼굴의 움직임이 시간적으로 동기화되었는지 평가하는 개선된 BFC 메트릭

    요소:
    - Event 기반 동기화: 큰 변화가 같은 타이밍에 일어났는가
    - 반응량 비율 penalty: 얼굴 반응이 너무 크거나 작으면 감점
    - Clip 단위 가중 평균

    Returns:
        float: 전체 평균 동기화 점수 (0~1, 높을수록 잘 동기화됨)
    """
    
    assert sum(clip_lengths) == len(vertex_sequence), "clip_lengths 합이 vertex_sequence 길이와 다름"
    assert sum(clip_lengths) - len(clip_lengths) == len(joint_velocities), "joint_velocities 길이와 맞지 않음"

    start_idx = 0
    sync_scores = []
    weights = []

    for length in clip_lengths:
        clip_vertex = vertex_sequence[start_idx:start_idx + length]      # (L, 50)
        clip_joints = joint_velocities[start_idx:start_idx + length - 1] # (L-1, J)

        # 얼굴 변화량: frame-to-frame displacement
        face_disp = np.linalg.norm(np.diff(clip_vertex, axis=0), axis=1)  # (L-1,)

        # 몸의 방향 변화량 (joint velocity 차이)
        joint_dir_change = np.linalg.norm(np.diff(clip_joints, axis=0), axis=1)  # (L-2,)

        # 얼굴 변화 이벤트 검출 (L-2,)
        face_events = face_disp[1:] > expression_threshold
        body_events = joint_dir_change > direction_threshold

        # 이벤트 동시 발생 비율
        min_len = min(len(face_events), len(body_events))
        face_events = face_events[:min_len]
        body_events = body_events[:min_len]

        sync_event_score = np.sum(face_events & body_events) / (np.sum(face_events | body_events) + 1e-6)

        # 얼굴:몸 평균 변화량 비율
        avg_face = np.mean(face_disp)
        avg_body = np.mean(np.linalg.norm(clip_joints, axis=1))
        ratio = avg_face / (avg_body + 1e-6)
        ratio_score = 1.0 / (1.0 + abs(ratio - target_ratio))  # 1.0이면 이상적, 벗어나면 줄어듦

        # 최종 score = 이벤트 동기화 * 비율 적정성
        final_score = sync_event_score * ratio_score
        sync_scores.append(final_score)
        weights.append(length)

        start_idx += length

    return np.average(sync_scores, weights=weights) if weights else 0




def get_BFC_metric(body_velocity_list, face_velocity_list, epsilon=1e-6):
    # Step 1: 전신 모션과 얼굴 모션의 변화량 크기
    body_norm = body_velocity_list.mean(axis=1) if body_velocity_list.ndim > 1 else body_velocity_list  # (frame 개수,)
    face_norm = np.linalg.norm(face_velocity_list, axis=-1).mean(axis=1)  # (frame 개수,)
    face_norm = face_norm * (np.mean(body_norm) / (np.mean(face_norm) + 1e-6))


    cosine_sim = np.dot(body_norm, face_norm) / (np.linalg.norm(body_norm) * np.linalg.norm(face_norm) + epsilon)

    # Step 2: 변화량의 상대적 차이를 사용하여 상관 계수 계산
    # if np.std(body_norm) > epsilon and np.std(face_norm) > epsilon:
    #     # 변화량을 각 프레임별 평균으로 정규화하여 비교
    #     normalized_body_norm = (body_norm - np.mean(body_norm)) / (np.std(body_norm) + epsilon)
    #     normalized_face_norm = (face_norm - np.mean(face_norm)) / (np.std(face_norm) + epsilon)
        
    #     # correlation_metric = np.corrcoef(normalized_body_norm, normalized_face_norm)[0, 1]
        
    # else:
    #     correlation_metric = 0  # 변동성이 없는 경우 0 반환

    # return correlation_metric
    return cosine_sim


import torch
import numpy as np

def expression_diversity_per_clip(vertex_list, clip_lengths):
    diversity_scores = []
    start_idx = 0

    
    for clip_length in clip_lengths:
        if clip_length == 0:
            # 빈 구간은 건너뛰기
            continue
        
        # 각 클립에 대한 vertex list 추출
        clip_vertex_list = vertex_list[start_idx:start_idx + clip_length]
        
        # 각 클립의 다양성 계산
        if clip_vertex_list.shape[0] > 0:  # 빈 클립이 아닌 경우에만 계산
            mean_vertices = clip_vertex_list.mean(dim=0) if isinstance(clip_vertex_list, torch.Tensor) else np.mean(clip_vertex_list, axis=0)
            deviations = torch.norm(clip_vertex_list - mean_vertices, dim=2) if isinstance(clip_vertex_list, torch.Tensor) else np.linalg.norm(clip_vertex_list - mean_vertices, axis=2)
            diversity_score = deviations.var().item() if isinstance(deviations, torch.Tensor) else np.var(deviations)
            diversity_scores.append(diversity_score)
        
        start_idx += clip_length
    
    # 전체 다양성 점수
    overall_diversity_score = np.mean(diversity_scores) if diversity_scores else 0
    
    return overall_diversity_score

def expression_diversity_weighted(vertex_list, clip_lengths):
    diversity_scores = []
    total_weight = sum(clip_lengths)  # 총 길이 (가중치 합)
    start_idx = 0
    
    for clip_length in clip_lengths:
        if clip_length == 0:
            # 빈 구간은 건너뛰기
            continue
        
        # 각 클립에 대한 vertex list 추출
        clip_vertex_list = vertex_list[start_idx:start_idx + clip_length]
        
        # 각 클립의 다양성 계산
        if clip_vertex_list.shape[0] > 0:
            mean_vertices = clip_vertex_list.mean(dim=0) if isinstance(clip_vertex_list, torch.Tensor) else np.mean(clip_vertex_list, axis=0)
            deviations = torch.norm(clip_vertex_list - mean_vertices, dim=2) if isinstance(clip_vertex_list, torch.Tensor) else np.linalg.norm(clip_vertex_list - mean_vertices, axis=2)
            diversity_score = deviations.var().item() if isinstance(deviations, torch.Tensor) else np.var(deviations)
            
            # 클립 길이에 따른 가중치 적용
            weighted_score = diversity_score * (clip_length / total_weight)
            diversity_scores.append(weighted_score)
        
        start_idx += clip_length
    
    # 가중치를 적용한 최종 다양성 점수
    overall_diversity_score = sum(diversity_scores)
    
    return overall_diversity_score


def calculate_reference_variance(vertex_list):
    if isinstance(vertex_list, torch.Tensor):
        ref_variance = vertex_list.var(dim=0)
    else:
        ref_variance = np.var(vertex_list, axis=0)
    
    # 자주 움직이는 vertex는 가중치를 낮추기 위해 역수로 변환
    ref_weights = 1 / (ref_variance + 1e-6)
    return ref_weights

def expression_diversity_absolute_change(vertex_list, clip_lengths):
    # 참조 가중치 계산
    ref_weights = calculate_reference_variance(vertex_list)
    diversity_scores = []
    total_weight = sum(clip_lengths)
    start_idx = 0

    for clip_length in clip_lengths:
        if clip_length == 0:
            continue
        
        # 각 클립에 대한 vertex list 추출
        clip_vertex_list = vertex_list[start_idx:start_idx + clip_length]
        
        if clip_vertex_list.shape[0] > 0:
            # 클립 내 모든 프레임 간 평균 변화량을 계산하여 절대적인 변화 평가
            diff = clip_vertex_list[1:] - clip_vertex_list[:-1]
            absolute_changes = (diff ** 2).mean(dim=0) if isinstance(diff, torch.Tensor) else np.mean(diff ** 2, axis=0)
            
            # 변화량에 가중치 적용
            weighted_changes = absolute_changes * ref_weights
            diversity_score = weighted_changes.mean().item() if isinstance(weighted_changes, torch.Tensor) else np.mean(weighted_changes)
            
            # 클립 길이에 따른 가중치 적용
            weighted_score = diversity_score * (clip_length / total_weight)
            diversity_scores.append(weighted_score)
        
        start_idx += clip_length
    
    # 가중치를 적용한 최종 다양성 점수
    overall_diversity_score = sum(diversity_scores)
    
    return overall_diversity_score

def expression_diversity_simple(vertex_list, clip_lengths):
    diversity_scores = []
    total_weight = sum(clip_lengths)
    start_idx = 0

    for clip_length in clip_lengths:
        if clip_length == 0:
            continue
        
        # 각 클립에 대한 vertex list 추출
        clip_vertex_list = vertex_list[start_idx:start_idx + clip_length]
        
        if clip_vertex_list.shape[0] > 1:  # 최소 2프레임 이상인 경우에만 계산
            # 각 클립 내 연속된 프레임 간 변화량 계산
            diff = clip_vertex_list[1:] - clip_vertex_list[:-1]
            absolute_changes = (diff ** 2).mean() if isinstance(diff, torch.Tensor) else np.mean(diff ** 2)
            
            diversity_score = absolute_changes.item() if isinstance(absolute_changes, torch.Tensor) else absolute_changes
            
            # 클립 길이에 따른 가중치 적용
            weighted_score = diversity_score * (clip_length / total_weight)
            diversity_scores.append(weighted_score)
        
        start_idx += clip_length
    
    # 가중치를 적용한 최종 다양성 점수
    overall_diversity_score = sum(diversity_scores)
    
    return overall_diversity_score

from sklearn.cluster import KMeans

def compute_naturalness_metric_per_clip(vertex_sequence, clip_lengths):
    start_idx = 0
    naturalness_scores = []
    weights = []
    
    for length in clip_lengths:
        # 현재 클립의 시퀀스 추출
        clip_sequence = vertex_sequence[start_idx:start_idx + length]
        
        # 프레임 간 변위 계산
        displacements = np.linalg.norm(np.diff(clip_sequence, axis=0), axis=-1)  # (clip_length-1, vertex 개수)
        mean_displacement_per_frame = displacements.mean(axis=1)  # 각 프레임의 평균 변위 (clip_length-1,)

        # 프레임 간 변위의 표준편차가 작을수록 자연스럽게 판단
        naturalness_score = 1 / (np.std(mean_displacement_per_frame) + 1e-6)  # 표준편차의 역수를 사용해 점수화
        naturalness_scores.append(naturalness_score)
        
        # 가중치를 클립 길이로 설정
        weights.append(length)
        
        # 다음 클립 시작 인덱스로 이동
        start_idx += length
    
    # 가중평균을 사용한 전체 자연스러움 점수 계산
    overall_naturalness_score = np.average(naturalness_scores, weights=weights)
    
    return overall_naturalness_score