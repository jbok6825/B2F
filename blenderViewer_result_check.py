

import sys
import os
import bpy
import torch
import atexit
import bpy, bmesh
import random
from mathutils import Vector


# from io_bvh import import_bvh


sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from process_dataset.Constant import * 
from SystemController import *
from SystemController_same import *
import utils_blender as utils_blender
from training.network.Util import *
from CharacterAnimationTools.anim import amass
from CharacterAnimationTools.anim import bvh

sys.path.append("/home/jbok6825/flame_to_arkit_project")

def update_camera():

    camera_orientation, camera_target_location = utils_blender.calculate_camera_info(new_armature_ours_sub, is_head=True)

    rv3d.view_location[0] = camera_target_location[0]
    rv3d.view_location[1] = camera_target_location[1]
    rv3d.view_location[2] = camera_target_location[2]

    rv3d.view_rotation = camera_orientation
    return 1/65


# def update_camera_with_animation():
#     # 현재 씬의 시작 프레임과 끝 프레임 가져오기
#     scene = bpy.context.scene
#     start_frame = scene.frame_start
#     end_frame = scene.frame_end

#     # 렌더링 카메라 가져오기 (없으면 생성)
#     camera = bpy.data.objects.get("Camera")
#     if camera is None:
#         # 카메라 생성
#         bpy.ops.object.camera_add(location=(0, 0, 0), rotation=(0, 0, 0))
#         camera = bpy.context.object
#         camera.name = "Camera"
#         print("New camera created!")

#     distance = 5.0  # 원하는 거리 (단위: meters)

#     # 지정된 프레임 범위에 대해 카메라 위치와 회전 업데이트 및 키프레임 삽입
#     for frame in range(start_frame, end_frame + 1):
#         scene.frame_set(frame)

#         # 카메라 정보 계산

#         camera_orientation, camera_target_location, face_direction = utils_blender.calculate_camera_info(new_armature, is_head=True)

#         # 카메라 위치 계산 (타겟 위치에서 일정 거리 뒤로 이동)
#         camera_location = camera_target_location + Vector([1, 0, 0]) * distance
#         camera.location = camera_location

#         # 카메라 회전 및 키프레임 삽입
#         camera.rotation_mode = 'QUATERNION'
#         if frame == 1:
#             camera.rotation_quaternion = camera_orientation
#         camera.keyframe_insert(data_path="location", frame=frame)

#         # camera.keyframe_insert(data_path="rotation_quaternion", frame=frame)

#     print(f"Keyframes inserted for frames {start_frame} to {end_frame}")





moving = False
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

        if event.type == 'M' and event.value == 'PRESS':
            global moving
            if moving == False:


                new_armature_ours.matrix_world.col[3][0] = 0.5
                new_armature_ours_sub.matrix_world.col[3][0] = 0.5
                new_armature_ours_noalign_sub.matrix_world.col[3][0] = -0.5
                # new_armature_ours_position.matrix_world.col[3][0] = 1.0
                # new_armature_ours_nocross.matrix_world.col[3][0] = 1.0
                # new_armature_ours_noconcon.matrix_world.col[3][0] = 1.5
                # new_armature_ours_nostylecon.matrix_world.col[3][0] = 2.0
                # new_armature_ours_noalign.matrix_world.col[3][0] = 2.5
                # new_armature_ours_same.matrix_world.col[3][0] = 3.0
                # new_armature_ours_smpl.matrix_world.col[3][0] = 3.5

                # new_armature_ours_2.matrix_world.col[3][0] = 1.5
                # 

                # new_armature_ours_sub.matrix_world.col[3][0] = 0.5
                # new_armature_ours_position_sub.matrix_world.col[3][0] = 1.0
                
                # new_armature_ours_noconcon_sub.matrix_world.col[3][0] = 1.5
                # new_armature_ours_nostylecon_sub.matrix_world.col[3][0] = 2.0
                # new_armature_ours_nocross_sub.matrix_world.col[3][0] = 2.5
                # new_armature_ours_same_sub.matrix_world.col[3][0] = 3.0
                # new_armature_ours_smpl_sub.matrix_world.col[3][0] = 3.5
                # new_armature_ours_noconsistnecy_sub.matrix_world.col[3][0] = 4.0



                moving = True
            else:
                new_armature_ours.matrix_world.col[3][0] = 0.
                # new_armature_ours_position.matrix_world.col[3][0] = 0.
                # new_armature_ours_nocross.matrix_world.col[3][0] = 0.
                # new_armature_ours_noconcon.matrix_world.col[3][0] = 0.
                # new_armature_ours_nostylecon.matrix_world.col[3][0] = 0.
                # new_armature_ours_noalign.matrix_world.col[3][0] = 0.
                # new_armature_ours_same.matrix_world.col[3][0] = 0.
                # new_armature_ours_smpl.matrix_world.col[3][0] = 0.

                # new_armature_ours_sub.matrix_world.col[3][0] = 0.0
                # new_armature_ours_position_sub.matrix_world.col[3][0] = 0.0
                # new_armature_ours_noconcon_sub.matrix_world.col[3][0] = 0.0
                # new_armature_ours_nostylecon_sub.matrix_world.col[3][0] = 0.0
                # new_armature_ours_nocross_sub.matrix_world.col[3][0] = 0.0
                # new_armature_ours_same_sub.matrix_world.col[3][0] = 0.0
                # new_armature_ours_smpl_sub.matrix_world.col[3][0] = 0.0
                # new_armature_ours_noconsistnecy_sub.matrix_world.col[3][0] = 0.0


                moving = False



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
    


utils_blender.remove_exist_objects()

for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        break
# 3d viewport의 정보 가져오기
space_data = area.spaces.active
rv3d = space_data.region_3d

common_path = "/home/jbok6825/dataset_MotionX/"
common_bvh_path = "/home/jbok6825/dataset_MotionX_bvh/"

## === emotion input facial animation == ##

# path_emotion_happy = common_path+"dance/subset_0000/A_Han_And_Tang_Dance_That_You_Will_Never_Get_Tired_Of.npy"
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

# path_model_flame_to_arkit = "/home/jbok6825/flame_to_arkit_project/Model/model_499epoch.pt" # 999epoch쯤이 결과 제일 자연스러움
# model_flame_to_arkit = torch.load(path_model_flame_to_arkit , map_location = DEVICE)
# for param_tensor in model_flame_to_arkit.state_dict():
#     model_flame_to_arkit.state_dict()[param_tensor] = model_flame_to_arkit.state_dict()[param_tensor].to(DEVICE)
# model_flame_to_arkit.eval()

## ==== test motion list ==== ##
# path_test_motion = "perform/subset_0000/Answer_Phone_clip_2.npy"

# path_test_motion = "perform/subset_0000/Dance_clip_12.npy"
# path_test_motion = "perform/subset_0000/Block_Sprinkler_clip_2.npy"
# path_test_motion = "idea400/subset_0029/Apology_Gesture_While_Standing.npy"

# path_test_motion = "idea400/subset_0016/Simultaneously_Neck_Pain_And_Standing.npy"
# path_test_motion = "dance/subset_0001/Dancing_Daily_Metoo_Zhang_Yuanying_clip_1.npy"
# path_test_motion = "perform/subset_0002/Throw_Stone.bvh"
# path_test_motion = "perform/subset_0002/Wake_Up_From_Sleep_clip_3.npy"
# path_test_motion = "fitness/subset_0016/Warrior_Iii_R_clip_23.npy"
# path_test_motion = "fitness/subset_0109/Warrior_Ii_Flow_R_clip_4.npy"

# ====확인된 test dataset=====
# path_test_motion = "idea400/subset_0019/Sneezing_While_Standing.npy"
# path_test_motion = "idea400/subset_0022/Simultaneously_Laundry_And_Standing.npy"
# path_test_motion = "kungfu/subset_0001/Kung_Fu_Flying_Kick_clip_1.npy"
# path_test_motion = "kungfu/subset_0001/Kung_Fu_Nunchucks_Training_Best_Nunchaku.npy"
# path_test_motion = "perform/subset_0002/Walk_Drunk_clip_7.npy"
# path_test_motion = "game_motion/subset_0034/Hand_Clapping_Choreography_clip_6.npy"
# path_test_motion = "fitness/subset_0045/Side_Reach_Up_Down_R_clip_24.npy"
# path_test_motion = "idea400/subset_0046/Walking_During_Origami_Airplane.npy"
# path_test_motion = "idea400/subset_0034/Standing_During_Circling_In_The_Air.npy"
# path_test_motion = "idea400/subset_0002/Sitting_During_Removing_The_Glasses.npy"
# path_test_motion = "dance/subset_0004/Swish_Swish_Is_Just_An_Ordinary_Blacksmith_clip_1.npy"
# path_test_motion = "fitness/subset_0001/Sport_Fitness_Standing_Left_And_Right_Leg_Lift_clip_6.npy"
# path_test_motion = "fitness/subset_0012/Side_Kick_To_Squat_R_clip_31.npy"
# path_test_motion = "idea400/subset_0026/Walking_During_Kicking_A_Soccer_Ball.npy"



# path_test_motion = "perform/subset_0002/Step_Into_Sauna.npy" # ground truth가 과장된듯 아닌듯
path_test_motion = "perform/subset_0002/Stir_Eggs_clip_3.npy" # 좀 정적이긴한데 괜찮은듯?
# path_test_motion = "perform/subset_0002/Stretch_clip_1.npy" # 얼굴 가리는 부분이 있긴 한데 ground truth 그때 이상하고 나머지 괜찮..
# path_test_motion = "perform/subset_0002/Take_Medicine.npy" # 괜찮긴한데 ours랑 ablation이랑 너무 비슷
# path_test_motion = "perform/subset_0002/Take_Medicine_clip_1.npy" # 괜찮음

# path_test_motion = "perform/subset_0002/Take_Shower_clip_4.npy" # 애매...ground truth는 괜찮은데 ablation이랑 ours랑 너무 비슷 (89epoch에서는 괜찮)



# path_test_motion = "perform/subset_0001/Post_Paper_clip_1.npy" # 차이가 드라마틱하진 않지만 괜찮
# path_test_motion = "perform/subset_0001/Pour_Liquor.npy" # 괜찮은듯
# path_test_motion = "perform/subset_0001/Pour_Water.npy" # 괜찮은듯
# path_test_motion = "perform/subset_0001/Pull_Neck.npy" # 괜찮은데 ablation이랑 차이가 넘 없음

# path_test_motion = "perform/subset_0001/Put_On_Earphones.npy" # ground truth 살짝 과장되었지만 괜찮음
# path_test_motion = "perform/subset_0001/Quarrel_clip_2.npy" # 차이가 좀 작긴하지만 괜찮음
# path_test_motion = "perform/subset_0001/Reach_Out_clip_3.npy" # 괜찮음

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
        
# test_22 좀 차이남
# test_9좀 차이남
# test_10 괜찮나..?
# test_11 길긴한데괜찮
# test_14, 15 좀 차이남(표정 차이가 젤 큼. 근데 이렇다고 젤 좋은 평가를 받을지는 잘 모르겠음)
#18 길긴한데 ours가 좀 결과가 ㄱㅊ


#===최종===#
# 17 750~900 frame 쓰면 될듯 + STYLE -6
# 19 250~400 frame 쓰면 될듯 + STYLE -6
# 11 350~500 frame 쓰면 될듯 + STYLE -6 (또는 100 ~250 근데 전신동작에서는 차이가 ..음)
# 12 550~700 frame 쓰면 될듯 + STYLE -6
# 13 90~240 frame 쓰면 될듯 + STYLE -6
# 16 100~250 frame 쓰면 될듯 + STYLE -6
# 20 320~470 frame 쓰면 될듯 + STYLE -6
# 21 140~290 frame 쓰면 될듯 + STYLE -6
# 22 100~250 frame 쓰면 될듯(동작 자체가 표정이 잘 안보이긴하지만)
# 24 490~640 frame 쓰면 될듯 + STYLE -6
# 29 + 550~700 frame
# catch case) 16 100~250 frame, catch case) 25 60~210frame

# style_motion_list[5][3][1][7] # style 7에서 ours가 강세
# 22+4 조합 ㄱㅊ 
index = 29
path_test_smpl_file = "/home/jbok6825/dataset_test/cmu/npy/test_"+str(index)+".npy"
path_test_bvh_file = "/home/jbok6825/dataset_test/cmu/bvh/test_"+str(index)+".bvh"

# path_test_smpl_file = "/home/jbok6825/dataset_MotionX/fitness/subset_0000/Sport_Fitness_Backward_Neck_Stretch.npy"


# style로 쓸만한거 
# 
# path_style_motion = "/home/jbok6825/dataset_MotionX/"+style_motion_list[-6] # 200~350 frame case1 
path_style_motion = "/home/jbok6825/dataset_MotionX/"+"perform/subset_0000/Answer_Phone_clip_2.npy" # case2
# path_style_motion = "/home/jbok6825/dataset_MotionX/"+"perform/subset_0002/Wake_Up_From_Sleep_clip_3.npy" # case3
# path_style_motion = "/home/jbok6825/dataset_MotionX/"+"fitness/subset_0001/Sport_Fitness_Jump_Up_And_Down_clip_16.npy" # case4
# path_style_motion = "/home/jbok6825/dataset_MotionX/perform/subset_0001/Perform_clip_1.npy" # case5
# path_style_motion = "/home/jbok6825/dataset_MotionX/HAA500/subset_0001/Baseball_Catch_Groundball.npy" # case6
# path_style_motion = common_path+"fitness/subset_0001/Sport_Fitness_Jump_Up_And_Down_clip_16.npy" # case7 (좀 약하긴함)
# path_style_motion = common_path + "perform/subset_0001/Drink_Cold_Water.npy" # case8(강력한 스타일 대비 차이는..)
#path_style_motion = "/home/jbok6825/dataset_MotionX/perform/subset_0001/Drag_Goods.npy" # case9
# path_style_motion = "/home/jbok6825/dataset_MotionX/perform/subset_0001/Fan.npy" # case10

# path_style_motion = "/home/jbok6825/dataset_MotionX/perform/subset_0001/Dish_Up.npy"




# path_style_motion = common_path + "perform/subset_0001/Drink_Cold_Water.npy" # catch1
# path_style_motion = "/home/jbok6825/dataset_MotionX/"+style_motion_list[-6]
# path_style_motion = "/home/jbok6825/dataset_MotionX/fitness/subset_0082/Hip_Lifts.npy"
# /home/jbok6825/dataset_MotionX/music/subset_0019/Play_The_Flute.npy
# /home/jbok6825/dataset_MotionX/fitness/subset_0086/Single_To_Double_Clap_Undef.npy 
# /home/jbok6825/dataset_MotionX/perform/subset_0001/Drink_Hot_Water.npy  




# path_style_motion = common_path+"fitness/subset_0013/Side_Stretch_R_clip_1.npy"
# path_style_motion = "/home/jbok6825/dataset_MotionX/"+style_motion_list[-5]
# path_style_motion = "/home/jbok6825/dataset_MotionX/"+style_motion_list[-16]
# path_style_motion = "/home/jbok6825/dataset_MotionX/"+style_motion_list[5]
# path_style_motion = "/home/jbok6825/dataset_MotionX/"+style_motion_list[-6]
# common_path+"fitness/subset_0013/Side_Stretch_R_clip_1.npy"

# /home/jbok6825/dataset_MotionX/perform/subset_0000/Call.npy

start_frame = 0

style_input_file = path_style_motion

utils_blender.load_animation_npz(path_test_smpl_file, "None", load_expression = False, load_bodymotion = True, start_frame = start_frame)
utils_blender.load_animation_npz(path_test_smpl_file, "Ours", load_expression = False, load_bodymotion = True, start_frame = start_frame)
# utils_blender.load_animation_npz(path_test_smpl_file, "Ours_NoPosition", load_expression = False, load_bodymotion = True, start_frame = start_frame)
# utils_blender.load_animation_npz(path_test_smpl_file, "Ours_NoCross", load_expression = False, load_bodymotion = True, start_frame = start_frame)
# utils_blender.load_animation_npz(path_test_smpl_file, "Ours_NoConCon", load_expression = False, load_bodymotion= True, start_frame = start_frame)
# utils_blender.load_animation_npz(path_test_smpl_file, "Ours_NoStyleCon", load_expression = False, load_bodymotion= True, start_frame = start_frame)
utils_blender.load_animation_npz(path_test_smpl_file, "Ours_NoAlign", load_expression = False, load_bodymotion= True, start_frame = start_frame)
# utils_blender.load_animation_npz(path_test_smpl_file, "Ours_SMPL", load_expression = False, load_bodymotion= True, start_frame = start_frame)
# utils_blender.load_animation_npz(path_test_smpl_file, "Ours_SAME", load_expression = False, load_bodymotion = True, start_frame = start_frame)
# utils_blender.load_animation_npz(path_test_smpl_file, "Ours_NoConsistnecy", load_expression = False, load_bodymotion = True, start_frame = start_frame)


utils_blender.load_animation_npz(style_input_file, "style_input", load_expression = True, load_bodymotion = False, start_frame = start_frame)
utils_blender.load_animation_npz(path_test_smpl_file, "Ours"+"_sub", load_expression = False, load_bodymotion= False, start_frame = start_frame)
# utils_blender.load_animation_npz(path_test_smpl_file, "Ours_64"+"_sub", load_expression = False, load_bodymotion= False, start_frame = start_frame)
# utils_blender.load_animation_npz(path_test_smpl_file, "Ours"+"_NoCross_sub", load_expression = False, load_bodymotion= False, start_frame = start_frame)
# # utils_blender.load_animation_npz(path_test_smpl_file, "Ours_NoConCon"+"_sub", load_expression = False, load_bodymotion= False, start_frame = start_frame)
# # utils_blender.load_animation_npz(path_test_smpl_file, "Ours_NoStyleCon_sub", load_expression = False, load_bodymotion= False, start_frame = start_frame)
utils_blender.load_animation_npz(path_test_smpl_file, "Ours_NoAlign_sub", load_expression = False, load_bodymotion= False, start_frame = start_frame)
# utils_blender.load_animation_npz(path_test_smpl_file, "Ours_SMPL"+"_sub", load_expression = False, load_bodymotion= False, start_frame = start_frame)
# utils_blender.load_animation_npz(path_test_smpl_file, "Ours_SAME"+"_sub", load_expression = False, load_bodymotion = False, start_frame = start_frame)
# utils_blender.load_animation_npz(path_test_smpl_file, "Ours_NoConsistnecy"+"_sub", load_expression = False, load_bodymotion = False, start_frame = start_frame)


new_armature= bpy.data.objects["None"]
style_armature = bpy.data.objects["style_input"]
print("##############", new_armature)

local_orientation = utils_blender.get_local_orientation_feature_from_smpl(new_armature)
global_position, global_orientation, global_velocity, character_local_coordinate = utils_blender.get_motion_feature_from_smpl(new_armature)


print(global_position.shape)
 

# # utils_blender.load_only_facial_animation(new_armature, path_style_motion)
new_armature_ours = bpy.data.objects["Ours"]
new_armature_ours_sub = bpy.data.objects["Ours_sub"]
# new_armature_ours_position = bpy.data.objects["Ours_NoPosition"]
# new_armature_ours_noconcon = bpy.data.objects["Ours_NoConCon"]
# new_armature_ours_nostylecon = bpy.data.objects["Ours_NoStyleCon"]
new_armature_ours_noalign = bpy.data.objects["Ours_NoAlign"]
# new_armature_ours_smpl = bpy.data.objects["Ours_SMPL"]
# new_armature_ours_same = bpy.data.objects["Ours_SAME"]
# new_armature_ours_noconsistnecy = bpy.data.objects["Ours_NoConsistnecy"]

# new_armature_none_sub = bpy.data.objects["None_sub"]

# new_armature_ours_position_sub = bpy.data.objects["Ours_NoPosition_sub"]
# new_armature_ours_noconcon_sub = bpy.data.objects["Ours_NoConCon_sub"]
# new_armature_ours_nostylecon_sub = bpy.data.objects["Ours_NoStyleCon_sub"]
new_armature_ours_noalign_sub = bpy.data.objects["Ours_NoAlign_sub"]
# new_armature_ours_smpl_sub = bpy.data.objects["Ours_SMPL_sub"]
# new_armature_ours_same_sub = bpy.data.objects["Ours_SAME_sub"]
# new_armature_ours_noconsistnecy_sub = bpy.data.objects["Ours_NoConsistnecy_sub"]



sc = SystemController(
                DEVICE,
                path_model = PATH_MODEL+"/Model_Ours/model_299epoch.pth",
                path_style_motion=style_input_file,
                # path_style_motion = path_emotion_happy,
                # path_sub_style_motion=path_emotion_contempt,
                # mode_two_style=True,
                # style_code = 0,
                # customDataset = dataset, 
                global_position = global_position,
                global_orientation = global_orientation,
                global_velocity = global_velocity,
                character_local_coordinate=character_local_coordinate,
                use_vae=True,
                use_normalize=True,
                positional_encoding=True,
                style_latent_dim=12*16)

sc_noalign = SystemController(
                DEVICE,
                path_model = PATH_MODEL+"/Model_Ours_NoAlign/model_299epoch.pth",
                path_style_motion=style_input_file,
                # path_style_motion = path_emotion_happy,
                # path_sub_style_motion=path_emotion_contempt,
                # mode_two_style=True,
                # style_code = 0,
                # customDataset = dataset, 
                global_position = global_position,
                global_orientation = global_orientation,
                global_velocity = global_velocity,
                character_local_coordinate=character_local_coordinate,
                use_vae=True,
                use_normalize=True,
                positional_encoding=True,
                style_latent_dim=12*16)





print("facial motion create complete")

jaw_motion, face_expr_motion = sc.create_total_facial_motion()
utils_blender.load_total_facial_animation(new_armature_ours, jaw_motion=jaw_motion, face_expr_motion=face_expr_motion)
utils_blender.load_total_facial_animation(new_armature_ours_sub, jaw_motion=jaw_motion, face_expr_motion=face_expr_motion)

jaw_motion, face_expr_motion = sc_noalign.create_total_facial_motion()
utils_blender.load_total_facial_animation(new_armature_ours_noalign, jaw_motion=jaw_motion, face_expr_motion=face_expr_motion)
utils_blender.load_total_facial_animation(new_armature_ours_noalign_sub, jaw_motion=jaw_motion, face_expr_motion=face_expr_motion)




bpy.utils.register_class(ModalOperator)
bpy.ops.object.modal_operator('INVOKE_DEFAULT')



