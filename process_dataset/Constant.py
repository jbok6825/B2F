PATH_DB_ORIGIN = "/home/jbok6825/dataset_MotionX"
PATH_DB_STYLE_ORIGIN = "/home/jbok6825/dataset_MotionX_style"
PATH_DB_BVH_ORIGIN = "home/jbok6825/dataset_MotionX_bvh"
NUM_PAST_MOTION = 10
NUM_PAST_FACE = 1
NUM_STYLE_FRAME = 90

import os
import sys
PATH_ROOT_DIR = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
PATH_DB = PATH_ROOT_DIR + "/dataset"
PATH_DB_STYLE = PATH_ROOT_DIR + "/dataset_style"
# PATH_CONTROLLER = "/home/jbok6825/FacialMotionSynthesisProject/Model_controller_past"+ str(NUM_PAST_MOTION)+"frame"
PATH_CONTROLLER = PATH_ROOT_DIR + "/Model_controller"
PATH_MODEL = PATH_ROOT_DIR + "/Model"



RUNTIME_EXTRACT_JOINT_LIST = [
              'pelvis', 
              'left_ankle', 
              'right_ankle', 
              'left_foot', 
              'right_foot', 
              'head', 
              'left_elbow', 
              'right_elbow', 
              'left_wrist', 
              'right_wrist',
              'left_index1',
              'right_index1']

DATASET_EXTRACT_JOINT_LIST = [
              'pelvis', 
              'left_ankle', 
              'right_ankle', 
              'left_foot', 
              'right_foot', 
              'head', 
              'left_elbow', 
              'right_elbow', 
              'left_wrist', 
              'right_wrist',
              'left_wrist',
              'right_wrist']



SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


SMPLX_JOINT_NAMES = [ # body joint
               'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 
                # jaw joint
               'jaw',
                # eye joint 
               'left_eye_smplhf', 'right_eye_smplhf', 
                # left hand joint
               'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3', 
                # right hand joint
               'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3']


DICT_STYLE_CODE = {
    "neutral": 0,
    "happy": 1, "happiness": 1,
    "disgust": 2, "disgusting": 2, "disgusted": 2,
    "surprise": 3, "surprising":3,
    "sad": 4, "sadness": 4,
    "angry":5, "anger":5,
    "fear":6,
    "contempt":7
}

STYLE_LABEL = ["neutral", "happy", "disgust", "surprise", "sad", "anger", "fear", "contempt"]

# 고유한 스타일 코드의 개수
NUM_STYLE_CODE= len(set(DICT_STYLE_CODE.values()))


ARKIT_BLENDSHAPE = [
    'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight',
    'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft',
    'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
    'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose',
    'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel',
    'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight',
    'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
    'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
    'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight'
]

TOON_GOON_ARKIT_TO_SHAPEKEY_MAP = {
    'browDownLeft': 'Brow_Drop_Left',
    'browDownRight': 'Brow_Drop_Right',
    'browInnerUp': 'Brow_Raise_Inner_Left',  # 또는 'Brow_Raise_Inner_Right'
    'browOuterUpLeft': 'Brow_Raise_Outer_Left',
    'browOuterUpRight': 'Brow_Raise_Outer_Right',
    'cheekPuff': 'Cheek_Blow_L',  # Cheeks_Suck도 가능
    'cheekSquintLeft': 'Cheek_Raise_L',
    'cheekSquintRight': 'Cheek_Raise_R',
    'eyeBlinkLeft': 'Eye_Blink_L',
    'eyeBlinkRight': 'Eye_Blink_R',
    'eyeLookDownLeft': 'Eye_Blink_L',  # 유사한 동작으로 처리
    'eyeLookDownRight': 'Eye_Blink_R',
    'eyeLookInLeft': 'Eye_Wide_L',
    'eyeLookInRight': 'Eye_Wide_R',
    'eyeLookOutLeft': 'Eye_Squint_L',
    'eyeLookOutRight': 'Eye_Squint_R',
    'eyeLookUpLeft': 'Eye_Blink_L',  # 중복 허용
    'eyeLookUpRight': 'Eye_Blink_R',
    'eyeSquintLeft': 'Eye_Squint_L',
    'eyeSquintRight': 'Eye_Squint_R',
    'eyeWideLeft': 'Eye_Wide_L',
    'eyeWideRight': 'Eye_Wide_R',
    'jawForward': 'Mouth_Lips_Jaw_Adjust',
    'jawLeft': 'Mouth_L',
    'jawOpen': 'Mouth_Open',
    'jawRight': 'Mouth_R',
    'mouthClose': 'Mouth_Pucker',  # 가장 유사한 동작으로 매핑
    'mouthDimpleLeft': 'Mouth_Dimple_L',
    'mouthDimpleRight': 'Mouth_Dimple_R',
    'mouthFrownLeft': 'Mouth_Frown_L',
    'mouthFrownRight': 'Mouth_Frown_R',
    'mouthFunnel': 'Mouth_Blow',  # 또는 'Mouth_Pucker_Open'
    'mouthLeft': 'Mouth_L',
    'mouthLowerDownLeft': 'Mouth_Bottom_Lip_Down',
    'mouthLowerDownRight': 'Mouth_Bottom_Lip_Down',
    'mouthPressLeft': 'Mouth_Lips_Tight',
    'mouthPressRight': 'Mouth_Lips_Tight',
    'mouthPucker': 'Mouth_Pucker',
    'mouthRight': 'Mouth_R',
    'mouthRollLower': 'Mouth_Bottom_Lip_Under',
    'mouthRollUpper': 'Mouth_Top_Lip_Under',
    'mouthShrugLower': 'Mouth_Bottom_Lip_Trans',
    'mouthShrugUpper': 'Mouth_Top_Lip_Up',  # 유사한 동작으로 처리
    'mouthSmileLeft': 'Mouth_Smile_L',
    'mouthSmileRight': 'Mouth_Smile_R',
    'mouthStretchLeft': 'Mouth_Widen',
    'mouthStretchRight': 'Mouth_Widen_Sides',
    'mouthUpperUpLeft': 'Mouth_Top_Lip_Up',
    'mouthUpperUpRight': 'Mouth_Top_Lip_Up',
    'noseSneerLeft': 'Mouth_Snarl_Upper_L',
    'noseSneerRight': 'Mouth_Snarl_Upper_R'
}

FEMALE_PADDING_ARKIT_TO_SHAPEKEY_MAP = {
    'browDownLeft': 'Brow_Drop_L',
    'browDownRight': 'Brow_Drop_R',
    'browInnerUp': 'Brow_Raise_Inner_L',  # 또는 'Brow_Raise_Inner_R'
    'browOuterUpLeft': 'Brow_Raise_Outer_L',
    'browOuterUpRight': 'Brow_Raise_Outer_R',
    'cheekPuff': 'Cheek_Puff_L',  # 또는 'Cheek_Puff_R'
    'cheekSquintLeft': 'Cheek_Raise_L',
    'cheekSquintRight': 'Cheek_Raise_R',
    'eyeBlinkLeft': 'Eye_Blink_L',
    'eyeBlinkRight': 'Eye_Blink_R',
    'eyeLookDownLeft': 'Eye_L_Look_Down',
    'eyeLookDownRight': 'Eye_R_Look_Down',
    'eyeLookInLeft': 'Eye_L_Look_R',
    'eyeLookInRight': 'Eye_R_Look_L',
    'eyeLookOutLeft': 'Eye_L_Look_L',
    'eyeLookOutRight': 'Eye_R_Look_R',
    'eyeLookUpLeft': 'Eye_L_Look_Up',
    'eyeLookUpRight': 'Eye_R_Look_Up',
    'eyeSquintLeft': 'Eye_Squint_L',
    'eyeSquintRight': 'Eye_Squint_R',
    'eyeWideLeft': 'Eye_Wide_L',
    'eyeWideRight': 'Eye_Wide_R',
    'jawForward': 'Jaw_Forward',
    'jawLeft': 'Jaw_L',
    'jawOpen': 'Jaw_Open',
    'jawRight': 'Jaw_R',
    'mouthClose': 'Mouth_Close',
    'mouthDimpleLeft': 'Mouth_Dimple_L',
    'mouthDimpleRight': 'Mouth_Dimple_R',
    'mouthFrownLeft': 'Mouth_Frown_L',
    'mouthFrownRight': 'Mouth_Frown_R',
    'mouthFunnel': 'Mouth_Funnel',
    'mouthLeft': 'Mouth_L',
    'mouthLowerDownLeft': 'Mouth_Down_Lower_L',
    'mouthLowerDownRight': 'Mouth_Down_Lower_R',
    'mouthPressLeft': 'Mouth_Press_L',
    'mouthPressRight': 'Mouth_Press_R',
    'mouthPucker': 'Mouth_Pucker',
    'mouthRight': 'Mouth_R',
    'mouthRollLower': 'Mouth_Roll_In_Lower',
    'mouthRollUpper': 'Mouth_Roll_In_Upper',
    'mouthShrugLower': 'Mouth_Shrug_Lower',
    'mouthShrugUpper': 'Mouth_Shrug_Upper',
    'mouthSmileLeft': 'Mouth_Smile_L',
    'mouthSmileRight': 'Mouth_Smile_R',
    'mouthStretchLeft': 'Mouth_Stretch_L',
    'mouthStretchRight': 'Mouth_Stretch_R',
    'mouthUpperUpLeft': 'Mouth_Up_Upper_L',
    'mouthUpperUpRight': 'Mouth_Up_Upper_R',
    'noseSneerLeft': 'Nose_Sneer_L',
    'noseSneerRight': 'Nose_Sneer_R'
}

FEMALE_PARTY_ARKIT_TO_SHAPEKEY_MAP  = {
    "browInnerUp": "Brow_Raise_Inner_Left",
    "browDownLeft": "Brow_Drop_Left",
    "browDownRight": "Brow_Drop_Right",
    "browOuterUpLeft": "Brow_Raise_Outer_Left",
    "browOuterUpRight": "Brow_Raise_Outer_Right",
    "eyeLookUpLeft": "A06_Eye_Look_Up_Left",
    "eyeLookUpRight": "A07_Eye_Look_Up_Right",
    "eyeLookDownLeft": "A08_Eye_Look_Down_Left",
    "eyeLookDownRight": "A09_Eye_Look_Down_Right",
    "eyeLookInLeft": "A11_Eye_Look_In_Left",
    "eyeLookInRight": "A12_Eye_Look_In_Right",
    "eyeLookOutLeft": "A10_Eye_Look_Out_Left",
    "eyeLookOutRight": "A13_Eye_Look_Out_Right",
    "eyeBlinkLeft": "Eye_Blink_L",
    "eyeBlinkRight": "Eye_Blink_R",
    "eyeSquintLeft": "Eye_Squint_L",
    "eyeSquintRight": "Eye_Squint_R",
    "eyeWideLeft": "Eye_Wide_L",
    "eyeWideRight": "Eye_Wide_R",
    "jawOpen": "V_Open",
    "jawForward": "A26_Jaw_Forward",
    "jawLeft": "A27_Jaw_Left",
    "jawRight": "A28_Jaw_Right",
    "mouthFunnel": "A29_Mouth_Funnel",
    "mouthPucker": "A30_Mouth_Pucker",
    "mouthLeft": "A31_Mouth_Left",
    "mouthRight": "A32_Mouth_Right",
    "mouthRollUpper": "A33_Mouth_Roll_Upper",
    "mouthRollLower": "A34_Mouth_Roll_Lower",
    "mouthShrugUpper": "A35_Mouth_Shrug_Upper",
    "mouthShrugLower": "A36_Mouth_Shrug_Lower",
    "mouthClose": "A37_Mouth_Close",
    "mouthSmileLeft": "Mouth_Smile_L",
    "mouthSmileRight": "Mouth_Smile_R",
    "mouthFrownLeft": "Mouth_Frown_L",
    "mouthFrownRight": "Mouth_Frown_R",
    "mouthDimpleLeft": "Mouth_Dimple_L",
    "mouthDimpleRight": "Mouth_Dimple_R",
    "mouthUpperUpLeft": "A44_Mouth_Upper_Up_Left",
    "mouthUpperUpRight": "A45_Mouth_Upper_Up_Right",
    "mouthLowerDownLeft": "A46_Mouth_Lower_Down_Left",
    "mouthLowerDownRight": "A47_Mouth_Lower_Down_Right",
    "mouthPressLeft": "A48_Mouth_Press_Left",
    "mouthPressRight": "A49_Mouth_Press_Right",
    "mouthStretchLeft": "A50_Mouth_Stretch_Left",
    "mouthStretchRight": "A51_Mouth_Stretch_Right",
    "tongueOut": "T10_Tongue_Bulge_Left",
    "cheekPuff": "A20_Cheek_Puff",
    "cheekSquintLeft": "A21_Cheek_Squint_Left",
    "cheekSquintRight": "A22_Cheek_Squint_Right",
    "noseSneerLeft": "A23_Nose_Sneer_Left",
    "noseSneerRight": "A24_Nose_Sneer_Right"
}


import torch

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
MEAN_STYLE_VECTOR_LIST= torch.tensor([
    [0.1114, -0.2240, 0.0166, 0.1240, -0.1074, 0.1557, 0.0120, 0.1351, -0.1302, 0.2534, -0.1349, -0.0203, 0.2597, -0.1099, 0.2451, 0.1245, -0.2587, -0.1254, -0.0987, 0.0310, -0.2687, 0.0762, -0.2388, 0.3301, -0.3292, 0.3032, 0.1586, -0.1390, -0.1395, -0.0353, 0.0725, -0.0680],
    [-0.2508, 0.0764, -0.0347, 0.1107, 0.3490, -0.0498, 0.2746, 0.2043, 0.0464, -0.1470, -0.1511, 0.0316, -0.0102, -0.2451, 0.2714, -0.0896, -0.2529, -0.2118, 0.1403, -0.0383, -0.3036, 0.2327, 0.0092, -0.0127, -0.0698, 0.0306, -0.1698, 0.3468, 0.1354, -0.1906, -0.0243, -0.0156],
    [-0.1513, -0.0443, 0.0741, -0.1696, -0.0816, 0.3032, 0.3108, -0.0398, -0.0062, -0.0577, 0.2212, -0.1009, -0.0449, 0.0266, 0.1008, -0.0052, -0.2066, -0.1058, -0.2726, 0.2997, -0.3142, -0.1086, -0.0630, 0.3050, -0.1381, -0.1498, 0.2435, -0.1930, 0.2501, 0.1931, -0.0692, -0.0247],
    [0.2394, -0.2005, -0.2456, -0.2499, -0.1927, 0.0406, 0.1166, 0.4678, -0.0564, -0.1363, 0.3410, -0.1467, 0.0541, 0.1230, 0.0132, -0.0811, -0.2003, -0.1404, 0.0890, -0.0542, -0.2222, 0.1732, -0.1679, -0.0094, -0.0388, -0.0021, 0.0870, 0.1719, 0.1007, 0.1356, 0.2066, -0.1896],
    [-0.0116, 0.0568, 0.2496, 0.2050, -0.2586, 0.0333, 0.1855, 0.3552, 0.0139, 0.0318, 0.1970, 0.0577, -0.0912, -0.0003, -0.2216, 0.2316, 0.1753, -0.1712, 0.0381, -0.0829, -0.1061, 0.0514, -0.1115, 0.1399, -0.1894, 0.1531, 0.1249, -0.0856, -0.2196, -0.0477, -0.3577, -0.3665],
    [-0.0498, 0.0169, -0.0219, 0.0160, -0.1355, 0.0252, 0.2225, 0.2839, -0.2257, 0.0002, 0.0072, -0.3035, 0.0826, -0.1426, 0.0190, 0.1234, 0.0472, 0.2585, 0.1815, -0.0130, -0.2647, -0.2763, -0.2118, 0.0663, -0.4678, 0.2647, 0.0601, 0.1319, 0.1447, -0.0540, 0.1706, 0.0283],
    [-0.1830, 0.0506, 0.2025, -0.2194, -0.0874, -0.0908, 0.3402, 0.2647, 0.0556, -0.0611, 0.0368, 0.0101, 0.0964, 0.2081, 0.1881, -0.0726, 0.1895, -0.0603, -0.0177, 0.3686, -0.2272, -0.0306, -0.0610, -0.0388, 0.2566, -0.3244, -0.2207, -0.2233, -0.1774, -0.1826, -0.0593, 0.0068],
    [0.2323, -0.0094, -0.3365, 0.2419, 0.0479, 0.0070, -0.1406, -0.0217, -0.1280, -0.0439, -0.3478, 0.2739, 0.2041, -0.1162, 0.1479, -0.2927, 0.0717, 0.0442, 0.1204, -0.0169, 0.1331, 0.1114, -0.3436, -0.2364, -0.0202, 0.3211, 0.1739, 0.0478, 0.0519, -0.0297, -0.0642, -0.0706]
], device=DEVICE)



