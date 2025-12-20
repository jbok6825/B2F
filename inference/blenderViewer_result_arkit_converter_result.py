
import sys
import os
import bpy
import torch


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from process_dataset.Constant import *
from inference import utils_blender

sys.path.append("/home/jbok6825/flame_to_arkit_project")

def update_camera():

    camera_orientation, camera_target_location = utils_blender.calculate_camera_info(arkit_armature_0, joint_name = "CC_Base_Head")

    rv3d.view_location[0] = camera_target_location[0]
    rv3d.view_location[1] = camera_target_location[1]
    rv3d.view_location[2] = camera_target_location[2]

    # rv3d.view_rotation = camera_orientation
    return 1/65


utils_blender.remove_exist_objects()

for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        break
# 3d viewport의 정보 가져오기
space_data = area.spaces.active
rv3d = space_data.region_3d


class ModalOperator(bpy.types.Operator):
    bl_idname = "object.modal_operator"

    bl_label = "Simple Modal Operator"

    def __init__(self):
        print("modal operator start")

    def __del__(self):
        print("modal operator end")

    def modal(self, context, event):

        if event.type == 'F':
            self.start_stop_timer(update_camera)
        # return {'RUNNING_MODAL'}
            
        return {'PASS_THROUGH'}
        
    def invoke(self, context, event):
        context.window_manager.modal_handler_add(self)

        return {'RUNNING_MODAL'}
    
    def start_stop_timer(self, func):
        if bpy.app.timers.is_registered(func):
            bpy.app.timers.unregister(func)
            print("Timer stopped")
        else:
            bpy.app.timers.register(func)
            print("Timer started")

## === emotion input facial animation == ##


style_motion_list = [
        "dance/subset_0001/Electrician_Version_Love_Shot_Dormitory_Million_Lights_clip_4.npy", # test set 맞음 happy,
        "dance/subset_0000/Bai_Jingting_Said_It_Looks_Good_And_Then_I_Posted_It_Clip1.npy", # test set 맞음 happy,
        "dance/subset_0000/A_Han_And_Tang_Dance_That_You_Will_Never_Get_Tired_Of_clip_1.npy", # test set 맞음 happy
        "perform/subset_0002/Stand_Hot_clip_1.npy", # test set 맞음 anger
        "fitness/subset_0003/Sport_Fitness_Lying_Down_Left_Ankle_Circles.npy", # test set맞음  disgust
        "perform/subset_0000/Cast_Stone.npy", # test set 맞음  disgust
        "dance/subset_0002/Say_No_To_Appearance_Anxiety_Allergy_G_I_Dle_clip_3.npy", # test set 맞음 sad 괜찮
        "game_motion/subset_0016/Katana_Sprungbeinschnitt_clip_7.npy", # test set 맞음 anger 괜찮
        "game_motion/subset_0016/Katana_Weed_clip_5.npy", # test set 맞음 neutral
        "game_motion/subset_0016/Katana_Weed_clip_8.npy", # test set 맞음 sad
        "game_motion/subset_0017/Katana_Berraschungsschwert.npy", # test set 맞음 neutral
        "game_motion/subset_0017/Kick_Leg.npy", # test set 맞음 surprise
        "game_motion/subset_0017/Kick_Round_clip_6.npy", # test set맞음 sad
        "game_motion/subset_0017/Kick_Spin_clip_4.npy", # test set 맞음 anger
        "game_motion/subset_0017/Kick_Stoppende.npy", # test set 맞음 sad
        "game_motion/subset_0018/Kick_Back_clip_9.npy", # test set 맞음 disgust
        "game_motion/subset_0018/Kick_Harter_clip_6.npy", # test set 맞음 anger
        "game_motion/subset_0018/Life_Tree_clip_4.npy", # test set 맞음 disgust
        "game_motion/subset_0020/Long_Weapon_Eingest_clip_10.npy", # test set 맞음 sad
        "humman/subset_0006/The_L_Word_Touch_My_Ankle.npy", # test set 맞음 happy
        "humman/subset_0008/Wei_Shi_S_Type_L_Dubbing_clip_6.npy", # test set 맞음 surprise 입을 모으는 스타일 체크 할때 좋을듯
        "humman/subset_0008/Warriors_Of_The_Second_Type_L.npy", # test set 맞음 입을 모으는 스타일 체크 할 때 좋을 듯
        "idea400/subset_0002/Cheer_Up_While_Standing_clip_1.npy", # test set 맞음 sad
        "idea400/subset_0003/Number_2_During_Walking.npy", # test set 맞음 happy
        "idea400/subset_0003/Sitting_During_Phone_Call_Gesture.npy", # test set맞음 happy
        "idea400/subset_0004/Lick_And_Sitting_At_The_Same_Time.npy", # test set 맞음 happy
        "idea400/subset_0004/Simultaneously_Wear_A_Mask_And_Standing.npy", # test set 맞음 neutral
        "idea400/subset_0004/Simultaneously_Wrist_Wrap_And_Sitting.npy", # test set 맞음 happy
        "idea400/subset_0004/Sitting_And_Laugh_At_The_Same_Time.npy", # test set 맞음 happy
        "idea400/subset_0004/Sitting_During_Laughing_Loudly.npy", # test set 맞음 happy
        "idea400/subset_0004/Sitting_While_Laugh.npy", # test set 맞음 happy
        "idea400/subset_0008/Simultaneously_Sitting_And_Blowing_Your_Nose.npy",# test set 맞음 surprise 좀 특징적이라 스타일체크할 때 좋을듯
        "idea400/subset_0008/Simultaneously_Sitting_And_Stretch,_Stretch_clip_1.npy", # test set 맞음 happy
        "idea400/subset_0008/Simultaneously_Sitting_And_Yawning.npy", # test set 맞음 happy
        "idea400/subset_0008/Simultaneously_Sitting_And_Yawning_clip_1.npy", # test set맞음 happy
        "idea400/subset_0008/Simultaneously_Standing_And_Side_Kick.npy", # test set 맞음 happy
        ]

# path_style_motion_0 = "/home/jbok6825/dataset_MotionX/"+style_motion_list[-6]
# path_style_motion_1 = "/home/jbok6825/dataset_MotionX/"+"idea400/subset_0004/Simultaneously_Wear_A_Mask_And_Standing.npy"
# path_style_motion_2 = "/home/jbok6825/dataset_MotionX/"+style_motion_list[11]

# converter 결과 보여줄 때 -5랑 -6이랑
# common_path+


common_path = "/home/jbok6825/dataset_MotionX/"
path_emotion_happy = common_path +"dance/subset_0000/Bai_Jingting_Said_It_Looks_Good_And_Then_I_Posted_It_Clip1.npy"
path_emotion_anger = common_path+"perform/subset_0000/Block_Sprinkler_clip_2.npy"
path_emotion_surprise = common_path+"animation/subset_0000/Ways_To_Catch_360.npy"
# path_emotion_surprise = "/home/jbok6825/dataset_MotionX/perform/subset_0000/Arrange_Hair.npy"
# path_emotion_neutral = common
# _path+"animation/subset_0000/Ways_To_Catch_Autograph.npy"
path_emotion_neutral = "/home/jbok6825/dataset_MotionX/music/subset_0015/Play_Pipa_clip_88.npy"
path_emotion_disgust = common_path+"perform/subset_0000/Answer_Phone_clip_2.npy"
# path_emotion_disgust = common_path + "perform/subset_0001/Drink_Cold_Water.npy"
path_emotion_sad = common_path+"perform/subset_0002/Wake_Up_From_Sleep_clip_3.npy"
path_emotion_fear = common_path+"fitness/subset_0013/Side_Stretch_R_clip_1.npy"
path_emotion_contempt = common_path+"kungfu/subset_0002/Play_Basketball_clip_4.npy"

path_style_motion_0 = common_path+"idea400/subset_0004/Simultaneously_Wear_A_Mask_And_Standing.npy"
# path_style_motion_0 = path_emotion_contempt # 잘 드러남 anger sad 안드러남
# path_style_motion_1 = path_emotion_surprise
# path_style_motion_2 = path_emotion_happyt

# path_emotion_fear = common_path+"fitness/subset_0013/Side_Stretch_R_clip_1.npy"
# path_emotion_contempt = common_path+
#vampire_break_door + style -7 조합 좋음
# 헐크 에어로빅 + style -6 조합 좋음

# fbx_file_path = "/home/jbok6825/robot.fbx"
# fbx_file_path = "/home/jbok6825/다운로드/Actorcore-Blender-1212-431003/Actor/toon-goon/toon-goon.fbx"
# fbx_skeleton_path = "/home/jbok6825/바탕화면/robot_arkit_retarget_3.bvh"
# fbx_skeleton_path = "/home/jbok6825/다운로드/customModel_xyz_fixed_sign.bvh"
# path_test_bvh_file = "/home/jbok6825/dataset_test/cmu/bvh/test_6.bvh"
# path_test_smpl_file = "/home/jbok6825/dataset_test/cmu/npy/test_6.npy"



def load_fbx(filepath):
    bpy.ops.import_scene.fbx(filepath=filepath)

def retarget_motion(mesh_object_name, motion_object_name):
    # Select the armature of the mesh
    mesh_armature = bpy.data.objects[mesh_object_name]
    motion_armature = bpy.data.objects[motion_object_name]

    if not mesh_armature or not motion_armature:
        print("Error: Armature not found.")
        return

    # Ensure both objects are armatures
    if mesh_armature.type != 'ARMATURE' or motion_armature.type != 'ARMATURE':
        print("Error: Both objects must be armatures.")
        return

    # Clear existing animation on the mesh armature
    mesh_armature.animation_data_clear()

    # Link the action from the motion armature to the mesh armature
    print("1) motion_armature.animation_data!!!!!!!!!!!!!!!!", motion_armature.animation_data)
    print("2) motion_armature.animation_data.action",motion_armature.animation_data.action)
    if motion_armature.animation_data and motion_armature.animation_data.action:
        mesh_armature.animation_data_create()
        mesh_armature.animation_data.action = motion_armature.animation_data.action
    else:
        print("Error: No animation data found on the motion armature.")
    
    # bpy.data.objects.remove(motion_armature, do_unlink=True)
    print("complete")

def insert_rest_pose_keyframe(armature_name):
    armature = bpy.data.objects.get(armature_name)

    if not armature or armature.type != 'ARMATURE':
        print(f"Error: Armature '{armature_name}' not found or is not an armature.")
        return

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')

    # Insert keyframes for all bones in the rest pose
    for bone in armature.pose.bones:
        bone.location = (0, 0, 0)
        bone.rotation_quaternion = (1, 0, 0, 0)
        bone.scale = (1, 1, 1)
        bone.keyframe_insert(data_path="location", frame=1)
        bone.keyframe_insert(data_path="rotation_quaternion", frame=1)
        bone.keyframe_insert(data_path="scale", frame=1)

    bpy.ops.object.mode_set(mode='OBJECT')
    print("Rest pose keyframe inserted.")

# Replace these paths with your FBX file paths
mesh_fbx_path = "/home/jbok6825/dataset_arkit/character/kid.fbx"
motion_fbx_path_1 = "/home/jbok6825/dataset_arkit/motion/kid/volleyball.fbx"
motion_fbx_path_2 = "/home/jbok6825/dataset_arkit/motion/kid/aerobic-dance.fbx"
motion_fbx_path_3 = "/home/jbok6825/dataset_arkit/motion/kid/locking.fbx"
path_test_bvh_motion_1= "/home/jbok6825/dataset_arkit/motion/smpl_retargeted/volleyball.bvh"
path_test_bvh_motion_2= "/home/jbok6825/dataset_arkit/motion/smpl_retargeted/aerobic-dance2.bvh"
path_test_bvh_motion_3= "/home/jbok6825/dataset_arkit/motion/smpl_retargeted/locking.bvh"

# Load the mesh with skeleton
load_fbx(mesh_fbx_path)


# Retarget the motion
target_object_name_0 = "Armature"  # Name of the armature from the mesh FBX


print("Motion retargeting complete!")



arkit_armature_0= bpy.data.objects["Armature"] ## fbx 바뀔때마다 수정 필요

utils_blender.load_animation_npz(path_style_motion_0, "Style_0", load_expression = True, load_bodymotion = False, start_frame = 0)



style_armature_0= bpy.data.objects["Style_0"] ## fbx 바뀔때마다 수정 필요



start_frame = 0

path_model_flame_to_arkit = "/home/jbok6825/flame_to_arkit_project/Model_moe_weight500/model_999epoch.pt"

model_flame_to_arkit = torch.load(path_model_flame_to_arkit , map_location = DEVICE)
for param_tensor in model_flame_to_arkit.state_dict():
    model_flame_to_arkit.state_dict()[param_tensor] = model_flame_to_arkit.state_dict()[param_tensor].to(DEVICE)
model_flame_to_arkit.eval()



utils_blender.retarget_flame_to_arkit(flame_armature=style_armature_0, arkit_armature=arkit_armature_0, model = model_flame_to_arkit, template = "person", arkit_mesh_name="Character")


bpy.utils.register_class(ModalOperator)
bpy.ops.object.modal_operator('INVOKE_DEFAULT')
