
import sys
import os
import bpy


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from process_dataset.Constant import *
from inference.SystemController import SystemController
from inference import utils_blender

sys.path.append("/home/jbok6825/flame_to_arkit_project")

def update_camera():

    camera_orientation, camera_target_location = utils_blender.calculate_camera_info(new_armature_ours, is_head=True)

    rv3d.view_location[0] = camera_target_location[0]
    rv3d.view_location[1] = camera_target_location[1]
    rv3d.view_location[2] = camera_target_location[2] - 0.1

    rv3d.view_rotation = camera_orientation
    return 1/65



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
                moving = True
            else:
                new_armature_ours.matrix_world.col[3][0] = 0.
                moving = False

            
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

path_emotion_happy = common_path +"dance/subset_0000/Bai_Jingting_Said_It_Looks_Good_And_Then_I_Posted_It_Clip1.npy"
path_emotion_anger = common_path+"perform/subset_0000/Block_Sprinkler_clip_2.npy"
path_emotion_disgust = common_path+"perform/subset_0000/Answer_Phone_clip_2.npy"
path_emotion_real_happy =common_path+ "idea400/subset_0004/Sitting_While_Laugh.npy"




path_emotion_surprise = common_path+"animation/subset_0000/Ways_To_Catch_360.npy"
path_emotion_neutral = "/home/jbok6825/dataset_MotionX/music/subset_0015/Play_Pipa_clip_88.npy"
path_emotion_sad = common_path+"perform/subset_0002/Wake_Up_From_Sleep_clip_3.npy"
path_emotion_fear = common_path+"fitness/subset_0013/Side_Stretch_R_clip_1.npy"
path_emotion_contempt = common_path+"kungfu/subset_0002/Play_Basketball_clip_4.npy"



'''
결과영상 various style1
    motion:31
    style_input_file = "/home/jbok6825/dataset_MotionX/"+"dance/subset_0000/A_Han_And_Tang_Dance_That_You_Will_Never_Get_Tired_Of_clip_1.npy" # case2
    

'''

style_input_file = path_emotion_real_happy
style_input_file_2 = path_emotion_contempt

index = 12

path_test_smpl_file = "/home/jbok6825/dataset_MotionX/perform/subset_0002/Throw_Stone.npy"


start_frame = 0


utils_blender.load_animation_npz(path_test_smpl_file, "None", load_expression = False, load_bodymotion = True, start_frame = start_frame)
utils_blender.load_animation_npz(path_test_smpl_file, "Ours", load_expression = False, load_bodymotion = True, start_frame = start_frame)

new_armature= bpy.data.objects["None"]
# style_armature = bpy.data.objects["style_input"]
print("##############", new_armature)

local_orientation = utils_blender.get_local_orientation_feature_from_smpl(new_armature)
global_position, global_orientation, global_velocity, character_local_coordinate = utils_blender.get_motion_feature_from_smpl(new_armature)


print(global_position.shape)

new_armature_ours = bpy.data.objects["Ours"]
sc = SystemController(
                DEVICE,
                path_model = PATH_MODEL+"/Model_Ours/model_299epoch.pth",
                path_style_motion=style_input_file,
                global_position = global_position,
                global_orientation = global_orientation,
                global_velocity = global_velocity,
                character_local_coordinate=character_local_coordinate,)


jaw_motion, face_expr_motion = sc.create_total_facial_motion()
utils_blender.load_total_facial_animation(new_armature_ours, jaw_motion=jaw_motion, face_expr_motion=face_expr_motion)

bpy.utils.register_class(ModalOperator)
bpy.ops.object.modal_operator('INVOKE_DEFAULT')
