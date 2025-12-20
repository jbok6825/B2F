import os
import sys

import bpy
import torch

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from process_dataset.Constant import *
from inference.SystemController import SystemController
from inference import utils_blender

# === User paths (edit these for your environment) ===
MESH_FBX_PATH = "/home/jbok6825/dataset_arkit/character/kid.fbx"
BODY_MOTION_BVH = "/home/jbok6825/dataset_arkit/motion/smpl_retargeted/volleyball.bvh"
STYLE_MOTION_NPY = "/home/jbok6825/dataset_MotionX/animation/subset_0000/Ways_To_Catch_360.npy"
FLAME_TO_ARKIT_MODEL = "/home/jbok6825/flame_to_arkit_project/Model_moe_weight500/model_999epoch.pt"

ARKIT_ARMATURE_NAME = "Armature"  # from the FBX mesh
SMPL_ARMATURE_NAME = "Ours"       # name used when loading the BVH below


def update_camera():
    camera_orientation, camera_target_location = utils_blender.calculate_camera_info(
        arkit_armature, joint_name="CC_Base_Head"
    )
    rv3d.view_location[0] = camera_target_location[0]
    rv3d.view_location[1] = camera_target_location[1]
    rv3d.view_location[2] = camera_target_location[2]
    rv3d.view_rotation = camera_orientation
    return 1 / 65


class ModalOperator(bpy.types.Operator):
    bl_idname = "object.modal_operator"
    bl_label = "Simple Modal Operator"

    def __del__(self):
        print("modal operator end")

    def modal(self, context, event):
        if event.type == "F":
            self.start_stop_timer(update_camera)
        return {"PASS_THROUGH"}

    def invoke(self, context, event):
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def start_stop_timer(self, func):
        if bpy.app.timers.is_registered(func):
            bpy.app.timers.unregister(func)
            print("Timer stopped")
        else:
            bpy.app.timers.register(func)
            print("Timer started")


# --- Scene prep ---
utils_blender.remove_exist_objects()
area = next(a for a in bpy.context.screen.areas if a.type == "VIEW_3D")
rv3d = area.spaces.active.region_3d

# --- Load assets ---
bpy.ops.import_scene.fbx(filepath=MESH_FBX_PATH)
utils_blender.load_animation_bvh(BODY_MOTION_BVH, SMPL_ARMATURE_NAME, channel="XYZ")

arkit_armature = bpy.data.objects[ARKIT_ARMATURE_NAME]
smpl_armature = bpy.data.objects[SMPL_ARMATURE_NAME]
# Also apply body motion directly to the FBX armature for body playback
utils_blender.load_bvh_animation_to_existing_character(BODY_MOTION_BVH, arkit_armature)

# --- Create facial motion with our model ---
global_position, global_orientation, global_velocity, character_local_coordinate = (
    utils_blender.get_motion_feature_from_smpl(smpl_armature)
)
sc = SystemController(
    DEVICE,
    path_model=f"{PATH_MODEL}/Model_Ours/model_299epoch.pth",
    path_style_motion=STYLE_MOTION_NPY,
    global_position=global_position,
    global_orientation=global_orientation,
    global_velocity=global_velocity,
    character_local_coordinate=character_local_coordinate,
)

jaw_motion, face_expr_motion = sc.create_total_facial_motion()
utils_blender.load_total_facial_animation(
    smpl_armature, jaw_motion=jaw_motion, face_expr_motion=face_expr_motion
)

# --- FLAME -> ARKit retarget ---
model_flame_to_arkit = torch.load(FLAME_TO_ARKIT_MODEL, map_location=DEVICE)
for param_tensor in model_flame_to_arkit.state_dict():
    model_flame_to_arkit.state_dict()[param_tensor] = model_flame_to_arkit.state_dict()[
        param_tensor
    ].to(DEVICE)
model_flame_to_arkit.eval()

utils_blender.retarget_flame_to_arkit(
    flame_armature=smpl_armature,
    arkit_armature=arkit_armature,
    model=model_flame_to_arkit,
    template="person",
    arkit_mesh_name="Character",
)

bpy.utils.register_class(ModalOperator)
bpy.ops.object.modal_operator("INVOKE_DEFAULT")
