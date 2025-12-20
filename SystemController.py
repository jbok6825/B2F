import numpy as np
import process_dataset.utils as utils
from process_dataset.Constant import *
from training.createTrainedB2FNetwork import create_expanded_network


class SystemController:
    """
    Minimal controller: loads the trained 'ours' network and produces facial motion
    from body motion (offline or runtime). All legacy SMPL/SAME/VAE variants removed.
    """

    def __init__(
        self,
        device,
        path_model=None,
        path_style_motion=None,
        mode_realTime=False,
        global_position=None,
        global_orientation=None,
        global_velocity=None,
        character_local_coordinate=None,
    ):
        self.device = device
        self.mode_realTime = mode_realTime
        self.mode_use_normalize = True
        self.positional_encoding = True
        self.style_latent_dim = 12 * 16

        if not self.mode_realTime:
            self.set_motion_data(
                global_position,
                global_orientation,
                global_velocity,
                character_local_coordinate,
            )

        if path_style_motion is None:
            raise ValueError("path_style_motion is required for facial style features.")
        self.load_style_motion(path_style_motion)

        self.load_model(path_model)

    def set_motion_data(
        self,
        global_position=None,
        global_orientation=None,
        global_velocity=None,
        character_local_coordinate=None,
    ):
        (
            self.current_character_local_position,
            self.current_character_local_orientation,
            self.current_character_local_velocity,
        ) = utils.get_motion_feature(
            global_position,
            global_orientation,
            global_velocity,
            character_local_coordinate,
            only_current=True,
        )

    def load_style_motion(self, file_path):
        motion_style = np.load(file_path)
        motion_parms_style = utils.parameterize_motionX(motion_style)
        self.facial_feature_style = utils.get_facial_feature(
            motion_parms_style
        )  # {'face_expr': ~, 'jaw':~}

    def load_model(self, path_model):
        self.model = create_expanded_network(
            path_preTrainedModel=path_model,
            use_normalize=self.mode_use_normalize,
            positional_encoding=self.positional_encoding,
            style_latent_dim=self.style_latent_dim,
            content_vae_mode=False,
        ).to(DEVICE)
        self.model.eval()

    def create_total_facial_motion(self):
        input_data = self.get_network_input()
        output = self.model(input_data, is_runtime=True)
        blendshape = output["blendshape_output"]
        return blendshape[0][:, :3], blendshape[0][:, 3:]

    def create_realTime_facial_motion(
        self, global_position, global_orientation, global_velocity, character_local_coordinate
    ):
        (
            self.current_character_local_position,
            self.current_character_local_orientation,
            self.current_character_local_velocity,
        ) = utils.get_motion_feature(
            global_position,
            global_orientation,
            global_velocity,
            character_local_coordinate,
            only_current=True,
        )

        if self.current_character_local_position.shape[0] > 90:
            self.current_character_local_position = self.current_character_local_position[-90:]
            self.current_character_local_orientation = self.current_character_local_orientation[-90:]
            self.current_character_local_velocity = self.current_character_local_velocity[-90:]

        input_data = self.get_network_input()
        blendshape = self.model(input_data, is_runtime=True)["blendshape_output"]
        return blendshape[0][-1, :3], blendshape[0][-1, 3:]

    def get_network_input(self):
        formatted_data = utils.get_formatted_data(
            position_feature=self.current_character_local_position,
            orientation_feature=self.current_character_local_orientation,
            velocity_feature=self.current_character_local_velocity,
            face_expr_style_feature=self.facial_feature_style["face_expr"],
            jaw_style_feature=self.facial_feature_style["jaw"],
        )

        return {
            "facial_motion_style": formatted_data["facial_style_feature"].unsqueeze(0),
            "body_motion_content": formatted_data["fullbody_feature"].unsqueeze(0),
        }
