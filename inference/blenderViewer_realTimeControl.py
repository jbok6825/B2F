import sys
import os
import bpy
import torch


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from process_dataset.Constant import *
from inference.SystemController import SystemController
from inference import utils_blender





class UpdateFacialMotionOperator(bpy.types.Operator):
    bl_idname = "object.modal_operator"

    bl_label = "Simple Modal Operator"


    def __init__(self):
        print("modal operator start")

    def __del__(self):
        print("modal operator end")

    def modal(self, context, event):
        # Check if the animation is playing
        current_frame = bpy.context.scene.frame_current
        if context.screen.is_animation_playing:

            self.start_timer(update_frame)  # Start the timer if it's not already running
        else:
            self.stop_timer(update_frame)

        return {'PASS_THROUGH'}
        
    def invoke(self, context, event):
        context.window_manager.modal_handler_add(self)

        return {'RUNNING_MODAL'}
    
    def start_timer(self, func):
        if bpy.app.timers.is_registered(func) == False:
            bpy.app.timers.register(func)

    def stop_timer(self, func):
        if bpy.app.timers.is_registered(func):
            bpy.app.timers.unregister(func)


utils_blender.remove_exist_objects()

for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        break
# 3d viewport의 정보 가져오기
space_data = area.spaces.active
rv3d = space_data.region_3d

common_path = "/home/jbok6825/dataset_MotionX/"
## === emotion input facial animation == ##

path_emotion_happy = common_path+"dance/subset_0000/A_Han_And_Tang_Dance_That_You_Will_Never_Get_Tired_Of.npy"
path_emotion_anger = common_path+"perform/subset_0000/Block_Sprinkler_clip_2.npy"
path_emotion_surprise = common_path+"animation/subset_0000/Ways_To_Catch_360.npy"
path_emotion_neutral = common_path+"animation/subset_0000/Ways_To_Catch_Autograph.npy"
path_emotion_disgust = common_path+"perform/subset_0000/Answer_Phone_clip_2.npy"
path_emotion_sad = common_path+"perform/subset_0002/Wake_Up_From_Sleep_clip_3.npy"
path_emotion_fear = common_path+"fitness/subset_0013/Side_Stretch_R_clip_1.npy"
path_emotion_contempt = common_path+"kungfu/subset_0002/Play_Basketball_clip_4.npy"

# path_model_flame_to_arkit = "/home/jbok6825/flame_to_arkit_project/Model/model_499epoch.pt" # 999epoch쯤이 결과 제일 자연스러움
# model_flame_to_arkit = torch.load(path_model_flame_to_arkit , map_location = DEVICE)
# for param_tensor in model_flame_to_arkit.state_dict():
#     model_flame_to_arkit.state_dict()[param_tensor] = model_flame_to_arkit.state_dict()[param_tensor].to(DEVICE)
# model_flame_to_arkit.eval()

## ==== test motion list ==== ##
# path_test_motion = "perform/subset_0000/Answer_Phone_clip_2.npy"
# path_test_motion = "perform/subset_0000/Block_Sprinkler_clip_2.npy"
# # path_test_motion = "animation/subset_0000/Ways_To_Catch_Autograph.npy"
# # path_test_motion = "perform/subset_0000/Dance_clip_12.npy"
# # path_test_motion = "idea400/subset_0029/Apology_Gesture_While_Standing.npy"
path_test_motion = "idea400/subset_0016/Simultaneously_Neck_Pain_And_Standing.npy"
# # path_test_motion = "dance/subset_0001/Dancing_Daily_Metoo_Zhang_Yuanying_clip_1.npy"
# # path_test_motion = "game_motion/subset_0034/Hand_Clapping_Choreography_clip_6.npy"
# path_test_motion = "perform/subset_0002/Throw_Stone.npy"


path_test_bvh_motion = "/home/jbok6825/FacialMotionSynthesisProject/itzy_same_skel.bvh"

utils_blender.load_animation_bvh(path_test_bvh_motion, STYLE_LABEL[1])


new_armature = bpy.data.objects[STYLE_LABEL[1]]

sc = SystemController(
                DEVICE,
                path_model = PATH_MODEL+"/Model_Ours/model_299epoch.pth",
                path_style_motion=path_emotion_happy,
                mode_realTime=True)
print("facial motion create complete")



from collections import deque

# 프레임별 누적 정보를 저장할 deque (초기화 시점에서 한 번만 실행)
global_position_stack = deque(maxlen=50)
global_orientation_stack = deque(maxlen=50)
global_velocity_stack = deque(maxlen=50)
character_local_coordinate_stack = deque(maxlen=50)

def update_frame():
    global global_position_stack
    global global_orientation_stack
    global global_velocity_stack
    global character_local_coordinate_stack
    global new_armature

    import time
    start_time = time.time()

    if not bpy.context.screen.is_animation_playing:
        return 0.033333

    if bpy.context.scene.frame_current == 1:
        global_position_stack.clear()
        global_orientation_stack.clear()
        global_velocity_stack.clear()
        character_local_coordinate_stack.clear()

        utils_blender.set_current_facial_motion(new_armature, torch.zeros((3,)), torch.zeros(50,))
        current_global_position, current_global_orientation, current_global_velocity, current_character_local_coordinate = utils_blender.get_current_motion_feature_from_smpl(
            new_armature, past_global_position=None)

    if len(global_position_stack) == 0:
        past_global_position = None
    else:
        past_global_position = global_position_stack[-1]

    current_global_position, current_global_orientation, current_global_velocity, current_character_local_coordinate = utils_blender.get_current_motion_feature_from_smpl(
        new_armature, past_global_position=past_global_position)

    # 슬라이딩 윈도우: 50프레임만 유지
    global_position_stack.append(current_global_position)
    global_orientation_stack.append(current_global_orientation)
    global_velocity_stack.append(current_global_velocity)
    character_local_coordinate_stack.append(current_character_local_coordinate)

    # 쌓인 길이가 충분할 때만 생성
    if len(global_position_stack) < 50:
        return 0.033333

    # 텐서로 변환 후 facial motion 생성
    pos_tensor = torch.cat(list(global_position_stack), dim=0)
    ori_tensor = torch.cat(list(global_orientation_stack), dim=0)
    vel_tensor = torch.cat(list(global_velocity_stack), dim=0)
    coord_tensor = torch.cat(list(character_local_coordinate_stack), dim=0)

    current_jaw_motion, current_expr_motion = sc.create_realTime_facial_motion(
        pos_tensor, ori_tensor, vel_tensor, coord_tensor)
    utils_blender.set_current_facial_motion(new_armature, current_jaw_motion, current_expr_motion)
    print(current_jaw_motion)

    end_time = time.time()
    elapsed = end_time - start_time
    fps = 1.0 / elapsed if elapsed > 0 else float('inf')

    if bpy.context.scene.frame_current % 10 == 0:
        mem_allocated = torch.cuda.memory_allocated() / 1024**2
        mem_reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"[Frame {bpy.context.scene.frame_current}] FPS: {fps:.2f}, Time: {elapsed:.4f}s")
        print(f"GPU Memory - Allocated: {mem_allocated:.1f} MB, Reserved: {mem_reserved:.1f} MB")

    # import gc
    # gc.collect()
    # torch.cuda.empty_cache()

    return 0.033333


# bpy.utils.register_class(ModalOperator)
# bpy.ops.object.modal_operator('INVOKE_DEFAULT')
bpy.utils.register_class(UpdateFacialMotionOperator)
bpy.ops.object.modal_operator('INVOKE_DEFAULT')
