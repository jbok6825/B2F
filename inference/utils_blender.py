import numpy as np
import bpy
import process_dataset.utils as utils
from process_dataset.Constant import * 
from CharacterAnimationTools.anim import amass
from CharacterAnimationTools.anim import bvh
from CharacterAnimationTools.util.quat import *
from scipy.spatial.transform import Rotation as R
import sys
from mathutils import Matrix, Euler, Quaternion, Vector
import math
from scipy.linalg import logm


sys.path.append('/home/jbok6825/.config/blender/4.1/scripts/addons/smplx_blender_addon')
print(sys.path)
import smplx_blender_addon
# print(sys.path[-1])
# from __init__ import *

# sys.path.append('/home/jbok6825/.config/blender/4.1/scripts/addons/smplx_blender_addon')
# from pose_utils import set_pose_from_rodrigues  # 정확한 파일 이름을 사용



EXP_SCALING_VALUE = 1

def remove_exist_objects():
    objs = bpy.data.objects
    for i in range(len(list(objs))):
        objs.remove(objs[0], do_unlink=True)

def get_character_pelvis_position(obj):
    '''
    blender 좌표 기준의 global position을 return
    '''
    pelvis_position = obj.pose.bones['pelvis'].location
    
    return np.array((pelvis_position[0], pelvis_position[2]* -1, pelvis_position[1]))

def get_character_local_coordinate(obj):
    '''
    character local coordinate의 경우 blender와 관계없이
    항상 root가 바라보는 방향쪽이 z축
    위쪽방향이 y축
    두 벡터를 외적한게 x축
    '''

    origin_position = get_character_pelvis_position(obj)

    z_axis_pelvis = np.array(obj.pose.bones['pelvis'].z_axis)
    z_axis_pelvis[2] = 0.

    vector_z_axis = z_axis_pelvis / np.linalg.norm(z_axis_pelvis)
    vector_y_axis = np.array([0., 0., 1.])
    vector_x_axis = np.cross(vector_y_axis, vector_z_axis)
    vector_x_axis = vector_x_axis / np.linalg.norm(vector_x_axis)

    character_local_coordinate = np.identity(4)

    character_local_coordinate[:3, 0] = vector_x_axis
    character_local_coordinate[:3, 1] = vector_y_axis
    character_local_coordinate[:3, 2] = vector_z_axis
    character_local_coordinate[:3, 3] = origin_position

    return character_local_coordinate

def get_joint_global_position(obj, bone_name):
    '''
    blender 좌표 기준의 global position을 return
    '''
    global_position = None

    if bone_name == 'pelvis':
        global_position = get_character_pelvis_position(obj)

    else:
        pb = obj.pose.bones[bone_name]
        global_position = obj.matrix_world @ pb.matrix @ pb.location

    return np.array(global_position)

def get_joint_character_local_position(obj, bone_name):
    
    global_position = [0., 0., 0., 1.]
    global_position[:3] = get_joint_global_position(obj, bone_name)
    character_local_coordinate = get_character_local_coordinate(obj)
    local_position = np.linalg.inv(character_local_coordinate) @ global_position

    return local_position[:3]

def get_jaw_pose_from_quat(quat):
    axis = quat.axis
    angle_rad = quat.angle

    return axis*angle_rad

def load_animation_to_existing_character(filepath, armature, load_expression=True, load_bodymotion=True, start_frame=0, frame_length = None):
    target_framerate = 30

    # .npz 파일 로드
    # print("Loading: " + filepath)
    data = utils.parameterize_motionX(np.load(filepath))
    if frame_length == None:
        trans = np.array(data["trans"][start_frame:].cpu())
        poses = np.array(data["poses"][start_frame:].cpu())
        expression = np.array(data["face_expr"][start_frame:].cpu())
    else:
        trans = np.array(data["trans"][start_frame:start_frame+frame_length].cpu())
        poses = np.array(data["poses"][start_frame:start_frame+frame_length].cpu())
        expression = np.array(data["face_expr"][start_frame:start_frame+frame_length].cpu())

    mocap_framerate = int(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else int(data["mocap_framerate"])
    betas = np.array(data["betas"].cpu())
    
    expression = np.clip(expression, -5, 5)

    # 캐릭터 객체 가져오기
    obj = armature.children[0]  # armature의 자식 객체가 얼굴 메쉬라고 가정

    # 장면 프레임 설정
    bpy.context.scene.render.fps = target_framerate
    bpy.context.scene.frame_start = 1

    # 기존 애니메이션 키프레임 삭제 (armature action의 fcurves 이용)
    if armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
        fcurves = action.fcurves

        # 각 bone의 rotation 관련 fcurve만 삭제
        for bone_name in armature.pose.bones.keys():
            for fcurve in fcurves:
                if fcurve.data_path.startswith(f'pose.bones["{bone_name}"].rotation_quaternion') or \
                   fcurve.data_path.startswith(f'pose.bones["{bone_name}"].location'):
                    fcurves.remove(fcurve)

    # 기존 shape key의 키프레임 삭제
    if obj.data.shape_keys and obj.data.shape_keys.animation_data:
        shape_action = obj.data.shape_keys.animation_data.action
        shape_fcurves = shape_action.fcurves

        # 각 shape key에 대해 fcurve를 찾아 삭제
        for expr_idx in range(expression.shape[1]):
            key_block_name = f"Exp{expr_idx:03}"
            data_path = f'key_blocks["{key_block_name}"].value'
            for fcurve in shape_fcurves:
                if fcurve.data_path == data_path:
                    shape_fcurves.remove(fcurve)

    # 새로운 Shape 키 설정
    bpy.ops.object.mode_set(mode='OBJECT')
    for index, beta in enumerate(betas):
        key_block_name = f"Shape{index:03}"
        if key_block_name in obj.data.shape_keys.key_blocks:
            obj.data.shape_keys.key_blocks[key_block_name].value = beta
        else:
            print(f"ERROR: No key block for: {key_block_name}")

    # 관절 위치 업데이트
    bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

    # 포즈 및 표현 키프레임 추가
    step_size = int(mocap_framerate / target_framerate)
    num_frames = trans.shape[0]
    num_keyframes = int(num_frames / step_size)

    # if len(bpy.data.actions) == 0:
    #     bpy.context.scene.frame_end = num_keyframes
    # elif num_keyframes > bpy.context.scene.frame_end:
    #     
    bpy.context.scene.frame_end = num_keyframes

    for index, frame in enumerate(range(0, num_frames, step_size)):
        # if (index % 100) == 0:
        #     print(f"  {index}/{num_keyframes}")
        current_frame = index + 1
        current_pose = poses[frame].reshape(-1, 3)
        current_trans = trans[frame]
        for idx, bone_name in enumerate(SMPLX_JOINT_NAMES):
            if bone_name == "pelvis":
                # 신체 이동 적용 (pelvis)
                if load_bodymotion:
                    armature.pose.bones[bone_name].location = Vector((current_trans[0], current_trans[1], current_trans[2]))
                    pose_rodrigues = current_pose[idx]
                    smplx_blender_addon.global_set_pose_from_rodrigues(armature, bone_name, pose_rodrigues.tolist())
                armature.pose.bones[bone_name].keyframe_insert('location', frame=current_frame)
                armature.pose.bones[bone_name].keyframe_insert('rotation_quaternion', frame=current_frame)

            elif bone_name == "jaw":
                if load_expression == True:
                    pose_rodrigues = current_pose[idx]
                    smplx_blender_addon.global_set_pose_from_rodrigues(armature, bone_name, pose_rodrigues.tolist())
                    armature.pose.bones[bone_name].keyframe_insert('rotation_quaternion', frame=current_frame)

            else:
                if load_bodymotion:
                    pose_rodrigues = current_pose[idx]
                    smplx_blender_addon.global_set_pose_from_rodrigues(armature, bone_name, pose_rodrigues.tolist())
                armature.pose.bones[bone_name].keyframe_insert('rotation_quaternion', frame=current_frame)

        # facial expression 설정 및 키프레임 추가
        if load_expression:
            for expr_idx in range(expression.shape[1]):
                current_expression = expression[frame].reshape(-1,)
                key_block_name = f"Exp{expr_idx:03}"
                obj.data.shape_keys.key_blocks[key_block_name].value = current_expression[expr_idx] * EXP_SCALING_VALUE
                obj.data.shape_keys.key_blocks[key_block_name].keyframe_insert('value', frame=current_frame)

    # print(f"  {num_keyframes}/{num_keyframes}")
    bpy.context.scene.frame_set(1)



def load_animation_npz(filepath, armature_name, load_expression = True, load_bodymotion = True, start_frame = 0, end_frame = None):
    target_framerate = 30

        # Load .npz file

    print("Loading: " + filepath)
    data = utils.parameterize_motionX(np.load(filepath))

    if end_frame == None:
        trans = np.array(data["trans"][start_frame:].cpu())
        gender = str(data["gender"])
        mocap_framerate = int(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else int(data["mocap_framerate"])
        betas = np.array(data["betas"].cpu())
        poses = np.array(data["poses"][start_frame:].cpu())
        expression = np.array(data["face_expr"][start_frame:].cpu())
        expression = np.clip(expression, -5, 5)
    else:
        trans = np.array(data["trans"][start_frame:end_frame+1].cpu())
        gender = str(data["gender"])
        mocap_framerate = int(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else int(data["mocap_framerate"])
        betas = np.array(data["betas"].cpu())
        poses = np.array(data["poses"][start_frame:end_frame+1].cpu())
        expression = np.array(data["face_expr"][start_frame:end_frame+1].cpu())
        expression = np.clip(expression, -5, 5)

   

    # np.savez("test_neutral.npz", trans=trans, gender=gender, mocap_frame_rate=mocap_framerate, betas=betas, poses=poses, expressions=expression)
    # np.savez("test_neutral.npz", trans=trans, gender="female", mocap_frame_rate=mocap_framerate, betas=betas, poses=poses, expressions=expression)


    if (bpy.context.active_object is not None):
        bpy.ops.object.mode_set(mode='OBJECT')

    # Add gender specific model
    bpy.context.window_manager.smplx_tool.smplx_gender = gender
    bpy.context.window_manager.smplx_tool.smplx_handpose = "flat"
    bpy.ops.scene.smplx_add_gender()

    obj = bpy.context.view_layer.objects.active
    armature = obj.parent

    # Append animation name to armature name
    # armature.name = armature.name + "_" + os.path.basename(filepath).replace(".npy", "")

    armature.name = armature_name

    bpy.context.scene.render.fps = target_framerate
    bpy.context.scene.frame_start = 1

    # Set shape and update joint locations
    bpy.ops.object.mode_set(mode='OBJECT')
    for index, beta in enumerate(betas):
        key_block_name = f"Shape{index:03}"

        if key_block_name in obj.data.shape_keys.key_blocks:
            obj.data.shape_keys.key_blocks[key_block_name].value = beta
        else:
            print(f"ERROR: No key block for: {key_block_name}")

    bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

    # Keyframe poses
    step_size = int(mocap_framerate / target_framerate)

    num_frames = trans.shape[0]
    num_keyframes = int(num_frames / step_size)


    if len(bpy.data.actions) == 0:
        # Set end frame if we don't have any previous animations in the scene
        bpy.context.scene.frame_end = num_keyframes
    elif num_keyframes > bpy.context.scene.frame_end:
        bpy.context.scene.frame_end = num_keyframes

    obj = bpy.context.view_layer.objects.active
    armature = obj.parent


    for index, frame in enumerate(range(0, num_frames, step_size)):
        # if (index % 100) == 0:
        #     print(f"  {index}/{num_keyframes}")
        current_frame = index + 1
        current_pose = poses[frame].reshape(-1, 3)
        current_trans = trans[frame]
        for index, bone_name in enumerate(SMPLX_JOINT_NAMES):
            if bone_name == "pelvis":
                # Keyframe pelvis location
                if load_bodymotion == True:
                    armature.pose.bones[bone_name].location = Vector((current_trans[0], current_trans[1], current_trans[2]))
                    pose_rodrigues = current_pose[index]
                    smplx_blender_addon.global_set_pose_from_rodrigues(armature = armature, bone_name = bone_name, rodrigues = pose_rodrigues)
                armature.pose.bones[bone_name].keyframe_insert('location', frame=current_frame)
                armature.pose.bones[bone_name].keyframe_insert('rotation_quaternion', frame=current_frame)


            elif bone_name == "jaw":
                if load_expression == True:
                    pose_rodrigues = current_pose[index]
                    smplx_blender_addon.global_set_pose_from_rodrigues(armature = armature, bone_name = bone_name, rodrigues = pose_rodrigues)
                    armature.pose.bones[bone_name].keyframe_insert('rotation_quaternion', frame=current_frame)

            else:
                if load_bodymotion == True:
                    pose_rodrigues = current_pose[index]
                    smplx_blender_addon.global_set_pose_from_rodrigues(armature = armature, bone_name = bone_name, rodrigues = pose_rodrigues)
                armature.pose.bones[bone_name].keyframe_insert('rotation_quaternion', frame=current_frame)

        if load_expression == True:
            for index in range(expression.shape[1]):
                current_expression = expression[frame].reshape(-1,)
                key_block_name = f"Exp{index:03}"
    
                obj.data.shape_keys.key_blocks[key_block_name].value = current_expression[index] * EXP_SCALING_VALUE
                obj.data.shape_keys.key_blocks[key_block_name].keyframe_insert('value', frame = current_frame)

    # print(f"  {num_keyframes}/{num_keyframes}")
    bpy.context.scene.frame_set(1)

def load_animation_npz_catch_trial(filepath, armature_name, load_expression=True, load_bodymotion=True, start_frame=0):
    target_framerate = 30

    # Load .npz file
    print("Loading: " + filepath)
    data = utils.parameterize_motionX(np.load(filepath))
    trans = np.array(data["trans"][start_frame:].cpu())
    gender = str(data["gender"])
    mocap_framerate = int(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else int(data["mocap_framerate"])
    betas = np.array(data["betas"].cpu())
    poses = np.array(data["poses"][start_frame:].cpu())
    expression = np.array(data["face_expr"][start_frame:].cpu())
    expression = np.clip(expression, -5, 5)

    if bpy.context.active_object is not None:
        bpy.ops.object.mode_set(mode='OBJECT')

    # Add gender-specific model
    bpy.context.window_manager.smplx_tool.smplx_gender = gender
    bpy.context.window_manager.smplx_tool.smplx_handpose = "flat"
    bpy.ops.scene.smplx_add_gender()

    obj = bpy.context.view_layer.objects.active
    armature = obj.parent

    armature.name = armature_name

    bpy.context.scene.render.fps = target_framerate
    bpy.context.scene.frame_start = 1

    # Set shape and update joint locations
    bpy.ops.object.mode_set(mode='OBJECT')
    for index, beta in enumerate(betas):
        key_block_name = f"Shape{index:03}"

        if key_block_name in obj.data.shape_keys.key_blocks:
            obj.data.shape_keys.key_blocks[key_block_name].value = beta
        else:
            print(f"ERROR: No key block for: {key_block_name}")

    bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

    # Keyframe poses with nearest neighbor interpolation
    step_size = int(mocap_framerate / target_framerate)
    num_frames = trans.shape[0]

    # Ensure the end frame is correctly set
    bpy.context.scene.frame_end = (num_frames // step_size) * step_size

    for frame in range(0, num_frames, step_size):
        current_frame = frame + 1
        current_pose = poses[frame].reshape(-1, 3)
        current_trans = trans[frame]

        for index, bone_name in enumerate(SMPLX_JOINT_NAMES):
            if bone_name == "pelvis" and load_bodymotion:
                armature.pose.bones[bone_name].location = Vector((current_trans[0], current_trans[1], current_trans[2]))
                pose_rodrigues = current_pose[index]
                smplx_blender_addon.global_set_pose_from_rodrigues(armature=armature, bone_name=bone_name, rodrigues=pose_rodrigues)
                armature.pose.bones[bone_name].keyframe_insert('location', frame=current_frame)

            if load_bodymotion and bone_name != "jaw":
                pose_rodrigues = current_pose[index]
                smplx_blender_addon.global_set_pose_from_rodrigues(armature=armature, bone_name=bone_name, rodrigues=pose_rodrigues)
                armature.pose.bones[bone_name].keyframe_insert('rotation_quaternion', frame=current_frame)

        if load_expression:
            for index in range(expression.shape[1]):
                current_expression = expression[frame].reshape(-1,)
                key_block_name = f"Exp{index:03}"
                obj.data.shape_keys.key_blocks[key_block_name].value = current_expression[index] * EXP_SCALING_VALUE
                obj.data.shape_keys.key_blocks[key_block_name].keyframe_insert('value', frame=current_frame)

    # Remove intermediate keyframes
    # Remove intermediate keyframes and enforce constant interpolation
    for fcurve in armature.animation_data.action.fcurves:
        keyframe_points = fcurve.keyframe_points
        keep_frames = set(range(1, bpy.context.scene.frame_end + 1, 30))  # 정확히 30 프레임 간격
        for i in range(len(keyframe_points) - 1, -1, -1):
            if keyframe_points[i].co[0] not in keep_frames:
                keyframe_points.remove(keyframe_points[i])

        # Set interpolation to Constant
        for keyframe in keyframe_points:
            keyframe.interpolation = 'CONSTANT'


    # Set interpolation to Constant (Nearest Neighbor-like behavior)
    for fcurve in armature.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = 'CONSTANT'

    bpy.context.scene.frame_set(1)
    print("Motion loaded with nearest neighbor interpolation.")


def load_only_facial_animation(armature, filepath, start_frame = 0, frame_num = 0):
    data = utils.parameterize_motionX(np.load(filepath))

    mocap_framerate = int(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else int(data["mocap_framerate"])
    betas = np.array(data["betas"].cpu())
    poses = np.array(data["poses"][start_frame:].cpu())
    expression = np.array(data["face_expr"][start_frame:].cpu())

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", expression.shape)

    while(expression.shape[0] < frame_num):
        expression = np.concatenate((expression, expression[::-1, :]), axis=0)
        poses = np.concatenate((poses, poses[::-1, :]), axis = 0)

    expression = np.clip(expression, -5, 5)
    obj = armature.children[0]  # armature의 자식 객체가 얼굴 메쉬라고 가정

    # 장면 프레임 설정
    bpy.context.scene.render.fps = mocap_framerate
    bpy.context.scene.frame_start = 1
    target_framerate = mocap_framerate




    # 새로운 Shape 키 설정
    bpy.ops.object.mode_set(mode='OBJECT')
    for index, beta in enumerate(betas):
        key_block_name = f"Shape{index:03}"
        if key_block_name in obj.data.shape_keys.key_blocks:
            obj.data.shape_keys.key_blocks[key_block_name].value = beta
        else:
            print(f"ERROR: No key block for: {key_block_name}")

    # 관절 위치 업데이트
    bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

    # 포즈 및 표현 키프레임 추가
    step_size = int(mocap_framerate / target_framerate)
    num_frames = expression.shape[0]

    # if len(bpy.data.actions) == 0:
    #     bpy.context.scene.frame_end = num_keyframes
    # elif num_keyframes > bpy.context.scene.frame_end:
    #     

    for index, frame in enumerate(range(0, num_frames, step_size)):
        current_frame = index + 1
        current_pose = poses[frame].reshape(-1, 3)
        for idx, bone_name in enumerate(SMPLX_JOINT_NAMES):
            if bone_name == "jaw":
                pose_rodrigues = current_pose[idx]
                smplx_blender_addon.global_set_pose_from_rodrigues(armature, bone_name, pose_rodrigues.tolist())
                armature.pose.bones[bone_name].keyframe_insert('rotation_quaternion', frame=current_frame)

        # facial expression 설정 및 키프레임 추가

        for expr_idx in range(expression.shape[1]):
            current_expression = expression[frame].reshape(-1,)
            key_block_name = f"Exp{expr_idx:03}"
            obj.data.shape_keys.key_blocks[key_block_name].value = current_expression[expr_idx] * EXP_SCALING_VALUE
            obj.data.shape_keys.key_blocks[key_block_name].keyframe_insert('value', frame=current_frame)

    # print(f"  {num_keyframes}/{num_keyframes}")
    bpy.context.scene.frame_set(1)



def load_animation_bvh(filepath_bvh, armature_name,  start_frame = 0, channel = "XYZ"):
    target_framerate = 30

        # Load .npz file
    filepath = "/home/jbok6825/FacialMotionSynthesisProject/example_smpl_file.npy"
    data = utils.parameterize_motionX(np.load(filepath))
    gender = str(data["gender"])
    mocap_framerate = int(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else int(data["mocap_framerate"])
    betas = np.array(data["betas"].cpu())


    # np.savez("test_neutral.npz", trans=trans, gender="female", mocap_frame_rate=mocap_framerate, betas=betas, poses=poses, expressions=expression)

    anim = bvh.load(filepath = filepath_bvh)
    joint_name_list_bvh = anim.joint_names
    trans = anim.positions[:, 0]/100
    poses = to_axis_angle(anim.quats)
    


    if (bpy.context.active_object is not None):
        bpy.ops.object.mode_set(mode='OBJECT')

    # Add gender specific model
    bpy.context.window_manager.smplx_tool.smplx_gender = gender
    bpy.context.window_manager.smplx_tool.smplx_handpose = "flat"
    bpy.ops.scene.smplx_add_gender()

    obj = bpy.context.view_layer.objects.active
    armature = obj.parent

    # Append animation name to armature name
    # armature.name = armature.name + "_" + os.path.basename(filepath).replace(".npy", "")

    armature.name = armature_name

    bpy.context.scene.render.fps = target_framerate
    bpy.context.scene.frame_start = 1

    # Set shape and update joint locations
    bpy.ops.object.mode_set(mode='OBJECT')
    for index, beta in enumerate(betas):
        key_block_name = f"Shape{index:03}"

        if key_block_name in obj.data.shape_keys.key_blocks:
            obj.data.shape_keys.key_blocks[key_block_name].value = beta
        else:
            print(f"ERROR: No key block for: {key_block_name}")

    bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

    # Keyframe poses
    step_size = int(mocap_framerate / target_framerate)

    num_frames = trans.shape[0]
    num_keyframes = int(num_frames / step_size)

    print(f"Adding pose keyframes: {num_keyframes}")

    if len(bpy.data.actions) == 0:
        # Set end frame if we don't have any previous animations in the scene
        bpy.context.scene.frame_end = num_keyframes
    elif num_keyframes > bpy.context.scene.frame_end:
        bpy.context.scene.frame_end = num_keyframes

    obj = bpy.context.view_layer.objects.active
    armature = obj.parent

    for index, frame in enumerate(range(0, num_frames, step_size)):
        if (index % 100) == 0:
            print(f"  {index}/{num_keyframes}")
        current_frame = index + 1
        current_pose = poses[frame].reshape(-1, 3)
        current_trans = trans[frame]
        for index, bone_name in enumerate(joint_name_list_bvh):
            if bone_name == "pelvis":
                # Keyframe pelvis location
                armature.pose.bones[bone_name].location = Vector((current_trans[0], current_trans[1]+0.4, current_trans[2]))
                armature.pose.bones[bone_name].keyframe_insert('location', frame=current_frame)
            pose_rodrigues = current_pose[index]
            ## channel에 따라 바꿔줘야함
            if channel == "XYZ":
                pose_rodrigues = Vector((pose_rodrigues[0], pose_rodrigues[1], pose_rodrigues[2]))
            elif channel == "ZXY":
                pose_rodrigues= Vector((pose_rodrigues[2], pose_rodrigues[0], pose_rodrigues[1]))
            elif channel == "ZYX":
                pose_rodrigues = Vector((pose_rodrigues[2], pose_rodrigues[1], pose_rodrigues[0]))
            else:
                exit()
            smplx_blender_addon.global_set_pose_from_rodrigues(armature = armature, bone_name = bone_name, rodrigues = pose_rodrigues)
            armature.pose.bones[bone_name].keyframe_insert('rotation_quaternion', frame=current_frame)

    print(f"  {num_keyframes}/{num_keyframes}")
    bpy.context.scene.frame_set(1)



def load_template():
    if (bpy.context.active_object is not None):
        bpy.ops.object.mode_set(mode='OBJECT')

    # Add gender specific model
    bpy.context.window_manager.smplx_tool.smplx_gender = "neutral"
    bpy.context.window_manager.smplx_tool.smplx_handpose = "flat"
    bpy.ops.scene.smplx_add_gender()

    obj = bpy.context.view_layer.objects.active
    armature = obj.parent

    template = np.load("/home/jbok6825/FacialMotionSynthesisProject/FLAME_ARKit_template/mat_final.npy")

    for index in range(len(template)):
        current_template = template[index]
        pose_rodrigues = current_template[100:103]
        current_frame = index + 1
        smplx_blender_addon.global_set_pose_from_rodrigues(armature, 'jaw', pose_rodrigues.tolist())
        armature.pose.bones['jaw'].keyframe_insert('rotation_quaternion', frame=current_frame)

        current_expression = current_template[:100]

        for index in range(100):
            key_block_name = f"Exp{index:03}"
            obj.data.shape_keys.key_blocks[key_block_name].value = current_expression[index] * EXP_SCALING_VALUE
            obj.data.shape_keys.key_blocks[key_block_name].keyframe_insert('value', frame = current_frame)


def load_total_facial_animation(armature, jaw_motion, face_expr_motion):

    # obj = bpy.context.view_layer.objects.active
    bpy.ops.object.mode_set(mode='OBJECT')
    obj = armature.children[0]
    num_frames = jaw_motion.shape[0]
    bpy.context.scene.frame_end = num_frames


    # 기존 jaw bone의 키프레임 삭제
    if armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
        fcurves = action.fcurves

        # 'jaw' bone의 rotation 관련 fcurve만 삭제
        for fcurve in fcurves:
            if fcurve.data_path.startswith('pose.bones["jaw"].rotation_quaternion'):
                fcurves.remove(fcurve)

    # 기존 shape key의 키프레임 삭제
    if obj.data.shape_keys and obj.data.shape_keys.animation_data:
        shape_action = obj.data.shape_keys.animation_data.action
        shape_fcurves = shape_action.fcurves

        # 각 shape key에 대해 fcurve를 찾아 삭제
        for expr_idx in range(face_expr_motion.shape[1]):
            key_block_name = f"Exp{expr_idx:03}"
            data_path = f'key_blocks["{key_block_name}"].value'
            for fcurve in shape_fcurves:
                if fcurve.data_path == data_path:
                    shape_fcurves.remove(fcurve)

    for index in range(num_frames):
        pose_rodrigues = jaw_motion[index]
        current_frame = index + 1
        smplx_blender_addon.global_set_pose_from_rodrigues(armature, 'jaw', pose_rodrigues.tolist())
        armature.pose.bones['jaw'].keyframe_insert('rotation_quaternion', frame=current_frame)

        current_expression = face_expr_motion[index].reshape(-1,)

        for index in range(face_expr_motion.shape[1]):
            key_block_name = f"Exp{index:03}"
            obj.data.shape_keys.key_blocks[key_block_name].value = current_expression[index] * EXP_SCALING_VALUE
            obj.data.shape_keys.key_blocks[key_block_name].keyframe_insert('value', frame = current_frame)

    bpy.context.scene.frame_set(1)

def load_total_facial_animtion_to_flame(armature, jaw_motion, face_expr_motion):

    bpy.ops.object.mode_set(mode='OBJECT')
    obj = armature.children[0]
    num_frames = jaw_motion.shape[0]

    for index in range(num_frames):
        pose_rodrigues = jaw_motion[index]
        current_frame = index + 1
        smplx_blender_addon.global_set_pose_from_rodrigues(armature, 'jaw', pose_rodrigues.tolist())
        armature.pose.bones['jaw'].keyframe_insert('rotation_quaternion', frame=current_frame)

        current_expression = face_expr_motion[index].reshape(-1,)

        for index in range(face_expr_motion.shape[1]):
            key_block_name = f"Exp{index:03}"
            armature.children[0].to_mesh().shape_keys[key_block_name].value = current_expression[index] * EXP_SCALING_VALUE
            armature.children[0].to_mesh().shape_keys[key_block_name].keyframe_insert('value', frame = current_frame)


def set_current_facial_motion(armature, jaw_motion, face_expr_motion):

    obj = armature.children[0]

    pose_rodrigues = jaw_motion
    print(jaw_motion)
    # bpy.context.scene.set_pose_from_rodrigues(armature, 'jaw', pose_rodrigues.tolist())
    smplx_blender_addon.global_set_pose_from_rodrigues(armature, 'jaw', pose_rodrigues.tolist())

    current_expression = face_expr_motion

    for index in range(face_expr_motion.shape[0]):
        key_block_name = f"Exp{index:03}"
        obj.data.shape_keys.key_blocks[key_block_name].value = current_expression[index] * EXP_SCALING_VALUE
        

def calculate_camera_info(target_armature, is_head = False, joint_name = None):
    
    if joint_name != None:
        target_bone = target_armature.pose.bones[joint_name]
    else:
        if is_head == False:
            target_bone = target_armature.pose.bones['pelvis']
        else:
            target_bone = target_armature.pose.bones['head']
    pelvis_position = target_armature.matrix_world @ target_bone.head

    camera_target_location = Vector(pelvis_position)
    

    root_orientation = target_armature.matrix_world.to_3x3() @ target_bone.matrix.to_3x3() @ target_bone.bone.matrix_local.to_3x3().inverted()
    root_orientation = Matrix(root_orientation).to_quaternion()
    angle = root_orientation.angle
    axis = root_orientation.axis
    rv = angle * axis
    rv[0] = 0.
    rv[1] = 0.
    axis = rv.normalized()
    angle_rad = rv.length

    camera_orientation = np.array(Quaternion(axis, angle_rad).to_matrix()) @ np.array(Euler((math.radians(85), 0.0, 0.0)).to_matrix())
    camera_orientation = Matrix(camera_orientation).to_quaternion()

    # head_orientation = target_armature.matrix_world.to_quaternion() @ target_bone.matrix.to_quaternion()

    # # 얼굴 방향 계산
    # face_direction = head_orientation @ Vector((0, 0, 1))  # 머리 본의 Z축이 얼굴 방향이라고 가정
    # face_direction[2] = 0.

    return camera_orientation, camera_target_location

def get_local_orientation_feature_from_smpl(armature):
    print("##################2", armature)
    action = armature.animation_data.action

    local_orientation = []

    for i in range(int(action.frame_range[0]), int(action.frame_range[1])+1):
        bpy.context.scene.frame_set(i)
        current_local_orientation = []

        for extract_joint in RUNTIME_EXTRACT_JOINT_LIST:
            posebone = armature.pose.bones[extract_joint]
            posebone_parent = posebone.parent

            global_matrix = armature.matrix_world.to_3x3() @ posebone.matrix.to_3x3() @ posebone.bone.matrix_local.to_3x3().inverted()
            global_orientation = utils.blender_matrix_to_opengl_matrix(global_matrix)

            global_matrix_parent = armature.matrix_world.to_3x3() @ posebone_parent.matrix.to_3x3() @ posebone_parent.bone.matrix_local.to_3x3().inverted()
            global_orientation_parent = utils.blender_matrix_to_opengl_matrix(global_matrix_parent)

            current_local_orientation.append(R.from_matrix(np.linalg.inv(global_orientation_parent) @ global_orientation).as_rotvec())

        local_orientation.append(torch.tensor(current_local_orientation, device = DEVICE, dtype = torch.float32))


    local_orientation = torch.stack(local_orientation, dim = 0)


    bpy.context.scene.frame_set(1)

    return local_orientation

def get_motion_feature_from_smpl(armature):
    action = armature.animation_data.action

    global_position = []
    global_orientation = []

    for i in range(int(action.frame_range[0]), int(action.frame_range[1])+1):
        bpy.context.scene.frame_set(i)
        current_global_position = []
        current_global_orientation  = []

        for extract_joint in RUNTIME_EXTRACT_JOINT_LIST:
            
            posebone = armature.pose.bones[extract_joint]
            current_global_position.append(utils.blender_position_to_opengl(posebone.head))
            global_matrix = armature.matrix_world.to_3x3() @ armature.pose.bones[extract_joint].matrix.to_3x3() @ posebone.bone.matrix_local.to_3x3().inverted()
            current_global_orientation.append(utils.blender_matrix_to_opengl_matrix(global_matrix))


        global_position.append(torch.tensor(current_global_position, device = DEVICE, dtype = torch.float32))
        global_orientation.append(torch.tensor(current_global_orientation, device = DEVICE, dtype = torch.float32))

    global_position = torch.stack(global_position, dim = 0)
    global_orientation= torch.stack(global_orientation, dim = 0)

    character_local_coordinate = utils.get_character_local_coordinate(global_position[:, 0], global_orientation[:, 0])
    global_velocity = torch.zeros_like(global_position)
    global_velocity[1:] = (global_position[1:] - global_position[:-1])*30
    global_velocity[0] = global_velocity[1] - (global_velocity[3] - global_velocity[2])

    global_position, global_orientation, global_velocity, character_local_coordinate

    bpy.context.scene.frame_set(1)

    return global_position, global_orientation, global_velocity, character_local_coordinate

def get_character_local_velocity_of_all_joint(armature):
    action = armature.animation_data.action

    global_position = []
    global_orientation = []

    for i in range(int(action.frame_range[0]), int(action.frame_range[1])+1):
        bpy.context.scene.frame_set(i)
        current_global_position = []
        current_global_orientation  = []

        for extract_joint in SMPL_JOINT_NAMES[1:]:
            
            posebone = armature.pose.bones[extract_joint]
            current_global_position.append(utils.blender_position_to_opengl(posebone.head))
            global_matrix = armature.matrix_world.to_3x3() @ armature.pose.bones[extract_joint].matrix.to_3x3() @ posebone.bone.matrix_local.to_3x3().inverted()
            current_global_orientation.append(utils.blender_matrix_to_opengl_matrix(global_matrix))

        global_position.append(torch.tensor(current_global_position, device = DEVICE, dtype = torch.float32))
        global_orientation.append(torch.tensor(current_global_orientation, device = DEVICE, dtype = torch.float32))

    global_position = torch.stack(global_position, dim = 0)
    global_orientation= torch.stack(global_orientation, dim = 0)
    character_local_coordinate = utils.get_character_local_coordinate(global_position[:, 0], global_orientation[:, 0])[:-1]
    global_velocity = (global_position[1:] - global_position[:-1])*30

    frame_num = len(global_velocity)
    joint_num = len(SMPL_JOINT_NAMES[1:])

    current_character_local_velocity = utils.get_local_coordinate_vector_value(torch.repeat_interleave(character_local_coordinate, joint_num, dim = 0), global_velocity.reshape(-1, 3)).reshape(frame_num, joint_num, 3)

 
    bpy.context.scene.frame_set(1)

    return current_character_local_velocity


import torch
import numpy as np

def get_velocity_and_angular_velocity_magnitude(armature, fps=30):
    action = armature.animation_data.action

    global_position = []
    global_orientation = []

    for i in range(int(action.frame_range[0]), int(action.frame_range[1]) + 1):
        bpy.context.scene.frame_set(i)
        current_global_position = []
        current_global_orientation = []

        for extract_joint in SMPL_JOINT_NAMES[1:]:  # root 제외
            posebone = armature.pose.bones[extract_joint]
            current_global_position.append(utils.blender_position_to_opengl(posebone.head))

            global_matrix = (
                armature.matrix_world.to_3x3() @
                armature.pose.bones[extract_joint].matrix.to_3x3() @
                posebone.bone.matrix_local.to_3x3().inverted()
            )
            current_global_orientation.append(utils.blender_matrix_to_opengl_matrix(global_matrix))

        global_position.append(torch.tensor(current_global_position, device=DEVICE, dtype=torch.float32))
        global_orientation.append(torch.tensor(current_global_orientation, device=DEVICE, dtype=torch.float32))

    global_position = torch.stack(global_position, dim=0)
    global_orientation = torch.stack(global_orientation, dim=0)

    # 선형 속도 크기 계산
    global_velocity = (global_position[1:] - global_position[:-1]) * fps  # 속도 계산
    velocity_magnitude = global_velocity.norm(dim=-1)  # 각 프레임과 관절에서 속도 크기 계산 (frame_num - 1, joint_num)

    # 각속도 크기 계산
    angular_velocity_magnitude = []
    for i in range(1, global_orientation.shape[0]):
        frame_angular_velocity_magnitude = []
        for j in range(len(SMPL_JOINT_NAMES[1:])):  # root 제외한 관절 개수
            # 두 회전 행렬 간의 상대 회전 계산
            rotation_change = global_orientation[i, j] @ global_orientation[i - 1, j].transpose(-1, -2)
            
            # Trace를 이용해 회전 각도(theta) 계산
            trace = rotation_change.diagonal(offset=0, dim1=-2, dim2=-1).sum()
            theta = torch.acos((trace - 1) / 2) * fps  # 회전 각도를 FPS에 맞춰 크기로 변환

            frame_angular_velocity_magnitude.append(theta)

        angular_velocity_magnitude.append(torch.stack(frame_angular_velocity_magnitude))

    angular_velocity_magnitude = torch.stack(angular_velocity_magnitude, dim=0)  # (frame_num - 1, joint_num)

    bpy.context.scene.frame_set(1)  # 프레임 초기화

    return velocity_magnitude, angular_velocity_magnitude

def get_current_motion_feature_from_smpl(armature, past_global_position):
    current_global_position = []
    current_global_orientation  = []

    for extract_joint in RUNTIME_EXTRACT_JOINT_LIST:
        posebone = armature.pose.bones[extract_joint]
        current_global_position.append(utils.blender_position_to_opengl(posebone.head))
        global_matrix = armature.matrix_world.to_3x3() @ armature.pose.bones[extract_joint].matrix.to_3x3() @ posebone.bone.matrix_local.to_3x3().inverted()
        current_global_orientation.append(utils.blender_matrix_to_opengl_matrix(global_matrix))

    current_global_position = torch.tensor(current_global_position, device = DEVICE, dtype=torch.float32)
    current_global_orientation = torch.tensor(current_global_orientation, device = DEVICE, dtype = torch.float32)

    current_character_local_coordinate = utils.get_character_local_coordinate(current_global_position[0].unsqueeze(0), current_global_orientation[0].unsqueeze(0)).squeeze(0)
    current_global_velocity = torch.zeros_like(current_global_position)
    
    if past_global_position != None:
        current_global_velocity = (current_global_position - past_global_position) * 30

    return current_global_position.unsqueeze(0), current_global_orientation.unsqueeze(0), current_global_velocity.unsqueeze(0), current_character_local_coordinate.unsqueeze(0)

def retarget_flame_to_arkit(flame_armature, arkit_armature, model, template, arkit_mesh_name= None):

    flame_mesh = flame_armature.children[0]
    flame_action = flame_armature.animation_data.action

    if arkit_mesh_name == None:
        if template == "toon":

            arkit_mesh = bpy.data.objects['CC_Game_Body'] # character바뀔때마다 수정해야함
        elif template == "person":
            arkit_mesh = arkit_armature.children[0]
        elif template == "person_girl":
            arkit_mesh = arkit_armature.children[0]
        else:
            arkit_mesh = bpy.data.objects['CC_Game_Body.001']
    else:
        arkit_mesh = bpy.data.objects[arkit_mesh_name]

    print(arkit_mesh)

    for i in range(int(flame_action.frame_range[0]), int(flame_action.frame_range[1])+1):
        current_frame = i
        vector_flame_expression = []
        bpy.context.scene.frame_set(i)
        for index in range(100):
            key_block_name = f"Exp{index:03}"
            vector_flame_expression.append(flame_mesh.data.shape_keys.key_blocks[key_block_name].value)
        vector_flame_expression += list(get_jaw_pose_from_quat(flame_armature.pose.bones['jaw'].rotation_quaternion))
        vector_arkit_expression = model.forward(torch.tensor(vector_flame_expression).unsqueeze(0).to(DEVICE)).squeeze(0)
        # print(vector_arkit_expression)

        weight = 1.3
        for index in range(len(ARKIT_BLENDSHAPE)):
            key_block_name = ARKIT_BLENDSHAPE[index]
            if template == "toon":

                weight = 1.5
                key_block_name = TOON_GOON_ARKIT_TO_SHAPEKEY_MAP.get(key_block_name)
            elif template == "person":
                key_block_name = FEMALE_PADDING_ARKIT_TO_SHAPEKEY_MAP.get(key_block_name)
                
            elif template == "person_girl":
                # key_block_name = FEMALE_PARTY_ARKIT_TO_SHAPEKEY_MAP.get(key_block_name)
                key_block_name = FEMALE_PADDING_ARKIT_TO_SHAPEKEY_MAP.get(key_block_name)
                weight = 2
                if key_block_name == "Mouth_Close":
                    weight = 0.5
            elif template == "magic":
                key_block_name = TOON_GOON_ARKIT_TO_SHAPEKEY_MAP.get(key_block_name)
                if key_block_name == "Mouth_Open":

                    arkit_mesh.data.shape_keys.key_blocks["V_Lip_Open"].value = vector_arkit_expression[index] * EXP_SCALING_VALUE * weight


            
            arkit_mesh.data.shape_keys.key_blocks[key_block_name].value = vector_arkit_expression[index] * EXP_SCALING_VALUE * weight
            
            

            arkit_mesh.data.shape_keys.key_blocks[key_block_name].keyframe_insert('value', frame = current_frame)
        bpy.context.scene.frame_set(1)


def retarget_arkit_to_flame(arkit_armature, flame_armature):

    arkit_mesh = arkit_armature.children[1]
    arkit_action = arkit_armature.animation_data.action
    
    template = np.load("/home/jbok6825/FacialMotionSynthesisProject/FLAME_ARKit_template/mat_final.npy")

    jaw_motion_list = []
    face_expr_list = []

    for i in range(int(arkit_action.frame_range[0]), int(arkit_action.frame_range[1])+1):
        vector_arkit_expression = []
        bpy.context.scene.frame_set(i)
        for index in range(51):
            key_block_name = ARKIT_BLENDSHAPE[index]
            vector_arkit_expression.append(arkit_mesh.data.shape_keys.key_blocks[key_block_name].value)

        vector_flame_expression = vector_arkit_expression @ template
        jaw_motion_list.append(vector_flame_expression[100:103])
        face_expr_list.append(vector_flame_expression[:100])

    load_total_facial_animation(flame_armature, np.array(jaw_motion_list), np.array(face_expr_list))



def get_vertex_positions_all_frames(obj, index_list, num_frames):
    # 결과 데이터를 저장할 리스트
    all_vertex_positions = []

    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_obj = obj.evaluated_get(depsgraph)
    
    # 각 프레임에 대해 메쉬 데이터를 읽어옴
    print("start")
    for frame in range(num_frames):
        bpy.context.scene.frame_set(frame + 1)  # 각 프레임 설정
        bpy.context.view_layer.update()  # 프레임 업데이트 최소화

        # 각 프레임의 vertex position을 numpy array로 저장
        mesh = evaluated_obj.to_mesh()
        vertex_positions = np.array([mesh.vertices[i].co[:] for i in index_list])


        all_vertex_positions.append(vertex_positions)
        # evaluated_obj.to_mesh_clear()

    return all_vertex_positions


def set_background():
    bpy.ops.mesh.primitive_plane_add(
    size=10,  # Plane 크기
    enter_editmode=False,  # 추가 후 Edit Mode로 진입 여부
    align='WORLD',  # 월드 좌표계 기준 정렬
    location=(0, 0, 0),  # 위치 (x, y, z)
    rotation=(0, 0, 0)  # 회전 (x, y, z)
    )

    bpy.ops.object.camera_add(
        enter_editmode=False,  # 추가 후 Edit Mode로 진입 여부
        align='WORLD',  # 월드 좌표 기준 정렬
        location=(0, -6.5, 0.9),  # 위치 (x, y, z)
        rotation=(np.deg2rad(90), 0, 0)  # 회전 (x, y, z) (라디안 단위)
    )


    if not bpy.data.worlds:
        world = bpy.data.worlds.new("NewWorld")
        bpy.context.scene.world = world
    else:
        world = bpy.context.scene.world

    # Use Nodes 활성화
    world.use_nodes = True

    # 노드 트리 가져오기
    nodes = world.node_tree.nodes

    # Background 노드 가져오기
    if "Background" not in nodes:
        background_node = nodes.new(type="ShaderNodeBackground")
    else:
        background_node = nodes["Background"]

    # 색상 설정
    value = 1.0
    background_node.inputs["Color"].default_value = (0.709, 0.709, 0.709, 1.0)  # (R, G, B, Alpha)

    # Strength 설정
    background_node.inputs["Strength"].default_value = 0.600

    # 출력 노드 연결 (확인 및 설정)
    output_node = nodes.get("World Output")
    if not output_node:
        output_node = nodes.new(type="ShaderNodeOutputWorld")
    world.node_tree.links.new(background_node.outputs["Background"], output_node.inputs["Surface"])


def load_bvh_animation_to_existing_character(filepath_bvh, armature):
    target_framerate = 30
    anim = bvh.load(filepath=filepath_bvh)
    joint_name_list_bvh = anim.joint_names
    trans = anim.positions[:, 0] / 100
    poses = to_axis_angle(anim.quats)

    root_offset = trans[0]
    trans -= root_offset

    bpy.context.scene.render.fps = target_framerate
    bpy.context.scene.frame_start = 1
    step_size = 1

    num_frames = trans.shape[0]
    num_keyframes = int(num_frames / step_size)
    bpy.context.scene.frame_end = max(bpy.context.scene.frame_end, num_keyframes)

    for index, frame in enumerate(range(0, num_frames, step_size)):
        current_frame = index + 1
        current_pose = poses[frame].reshape(-1, 3)
        current_trans = trans[frame]

        for idx, bone_name in enumerate(joint_name_list_bvh):
            if bone_name == joint_name_list_bvh[0]:
                armature.pose.bones[bone_name].location = Vector((current_trans[0], current_trans[1], current_trans[2]))
                armature.pose.bones[bone_name].keyframe_insert('location', frame=current_frame)
            pose_rodrigues = current_pose[idx]
            
            # 축 변환 적용
            # if "Leg" in bone_name or "Foot" in bone_name or  "Toe" in bone_name:
            # pose_rodrigues = convert_bvh_to_blender_axes(pose_rodrigues)
            
            smplx_blender_addon.global_set_pose_from_rodrigues(armature=armature, bone_name=bone_name, rodrigues=pose_rodrigues)
            armature.pose.bones[bone_name].keyframe_insert('rotation_quaternion', frame=current_frame)

        print(f"Frame {current_frame}, Bone: {bone_name}, Rodrigues: {pose_rodrigues}, Location: {current_trans}")
    print(f"  {num_keyframes}/{num_keyframes}")
    bpy.context.scene.frame_set(1)



    
# def convert_bvh_to_blender_axes(rodrigues):
#     # BVH -> Blender 축 변환 (Z-up 기준 예제)
#     return Vector((rodrigues[2], rodrigues[0], rodrigues[1]))