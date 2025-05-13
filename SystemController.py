import numpy as np
import process_dataset.utils as utils
import torch
import training.network as network
from process_dataset.Constant import * 
from training.createTrainedNetwork import create_expanded_network, create_smpl_version_network, create_diffuse_network
from diffusers import DDPMScheduler
class SystemController:
    def __init__(self, 
                 device, 
                 path_style_motion = None, #Motion-X data
                 path_sub_style_motion = None,
                 path_model = None, 
                 mode_realTime= False,
                 mode_smpl = False,
                 mode_two_style = False,
                 style_code = None,
                 local_orientation = None,
                 global_position = None, global_orientation = None, global_velocity = None, character_local_coordinate = None,
                 use_vae = False,
                 use_normalize = True,
                 positional_encoding = False,
                 style_latent_dim = 128):
        
        self.device = device
        self.mode_realTime = mode_realTime
        self.style_embdding_vector = None
        self.mode_smpl = mode_smpl
        self.mode_two_style = mode_two_style
        self.mode_use_vae = use_vae
        self.mode_use_normalize = use_normalize
        self.positional_encoding = positional_encoding
        self.style_latent_dim = style_latent_dim

        if self.mode_realTime == False:
            if mode_smpl == False:
                self.set_motion_data(global_position, global_orientation, global_velocity, character_local_coordinate)
            else:
                self.parent_local_orientation = local_orientation

        if path_style_motion != None:
            self.load_style_motion(path_style_motion, path_sub_style_motion)
        elif style_code != None:
            self.style_embdding_vector = MEAN_STYLE_VECTOR_LIST[style_code]
        
        self.load_model(path_model)
        model = torch.load(path_model, map_location='cpu')  # 또는 DEVICE
        print(model.state_dict().keys())


    def set_motion_data(self, global_position, global_orientation, global_velocity, character_local_coordinate):
        self.current_character_local_position, self.current_character_local_orientation, self.current_character_local_velocity = utils.get_motion_feature(global_position, global_orientation, global_velocity, character_local_coordinate, only_current=True)

    def load_style_motion(self, file_path, sub_file_path = None):
        
        if self.mode_two_style == False:
            self.motion_style = np.load(file_path)
            self.motion_parms_style = utils.parameterize_motionX(self.motion_style)
            self.frame_length_style = utils.get_frame_legnth(self.motion_parms_style)
            self.facial_feature_style = utils.get_facial_feature(self.motion_parms_style)  # {'face_expr': ~, 'jaw':~}
        else:
            self.motion_style_0 = np.load(file_path)
            self.motion_parms_style_0 = utils.parameterize_motionX(self.motion_style_0)
            self.frame_length_style_0 = utils.get_frame_legnth(self.motion_parms_style_0)
            self.facial_feature_style_0 = utils.get_facial_feature(self.motion_parms_style_0)  # {'face_expr': ~, 'jaw':~}

            self.motion_style_1 = np.load(sub_file_path)
            self.motion_parms_style_1 = utils.parameterize_motionX(self.motion_style_1)
            self.frame_length_style_1 = utils.get_frame_legnth(self.motion_parms_style_1)
            self.facial_feature_style_1 = utils.get_facial_feature(self.motion_parms_style_1)  # {'face_expr': ~, 'jaw':~}

    def set_style_motion(self, style_code):

        sample_data = self.customDataset.sample_style_data(style_code)
        self.facial_feature_style = {
            'face_expr': sample_data[:, 3:],
            'jaw': sample_data[:, :3]
        }

    def reset_style_data(self, style_code = None, file_path = None):

        if style_code != None:
            self.style_embdding_vector = MEAN_STYLE_VECTOR_LIST[style_code]
        else:
            self.load_style_motion(file_path)

    def load_model(self, path_model):
        if self.mode_smpl == False:
            self.model= create_expanded_network(path_preTrainedModel = path_model, style_latent_dim=self.style_latent_dim, content_vae_mode=False).to(DEVICE)
            # self.model= create_diffuse_network(path_preTrainedModel = path_model).to(DEVICE)
        else:
            self.model= create_smpl_version_network(path_preTrainedModel = path_model).to(DEVICE)
        self.model.eval()


    def create_total_facial_motion(self, interpolation = False, ratio = None):

        input_data = self.get_network_input()
        style_feature = input_data['facial_motion_style']
        body_motion_feature = input_data['body_motion_content']
        

        if interpolation == False:
            print(input_data)
            blendshape = self.model(input_data, is_runtime=True)['blendshape_output']


        else:
            if ratio == None:
                blendshape = self.model.forward_with_two_style(input_data, interpolation=True)['blendshape_output']
            else:
                blendshape = self.model.forward_with_two_style(input_data, interpolation=False, ratio = ratio)['blendshape_output']



        return blendshape[0][:, :3], blendshape[0][ :, 3:]
    
    def create_realTime_facial_motion(self, global_position, global_orientation, global_velocity, character_local_coordinate):

        self.current_character_local_position, self.current_character_local_orientation, self.current_character_local_velocity = utils.get_motion_feature(global_position, global_orientation, global_velocity, character_local_coordinate, only_current=True)
        
        if self.current_character_local_position.shape[0] > 90:
            self.current_character_local_position = self.current_character_local_position[-90:]
            self.current_character_local_orientation = self.current_character_local_orientation[-90:]
            self.current_character_local_velocity = self.current_character_local_velocity[-90:]

        input_data = self.get_network_input()
        blendshape= self.model(input_data, is_runtime=True)['blendshape_output']


        return blendshape[0][-1, :3], blendshape[0][-1, 3:]
        # return blendshape[0][:, :3], blendshape[0][:, 3:]

    def get_network_input(self, ):

        if self.style_embdding_vector == None:
            if self.mode_smpl == False:
                if self.mode_two_style == False:
                    formatted_data = utils.get_formatted_data( 
                                        position_feature = self.current_character_local_position,
                                        orientation_feature = self.current_character_local_orientation,
                                        velocity_feature = self.current_character_local_velocity,
                                        face_expr_style_feature = self.facial_feature_style['face_expr'],
                                        jaw_style_feature = self.facial_feature_style['jaw'])

                    return {
                        'facial_motion_style': formatted_data['facial_style_feature'].unsqueeze(0),
                        'body_motion_content': formatted_data['fullbody_feature'].unsqueeze(0),
                    }
                else:
                    formatted_data = utils.get_formatted_data( 
                                        position_feature = self.current_character_local_position,
                                        orientation_feature = self.current_character_local_orientation,
                                        velocity_feature = self.current_character_local_velocity)
                    
                    facial_style_feature_0 = torch.cat((self.facial_feature_style_0['jaw'], self.facial_feature_style_0['face_expr']), dim = -1).to(DEVICE)
                    facial_style_feature_1 = torch.cat((self.facial_feature_style_1['jaw'], self.facial_feature_style_1['face_expr']), dim = -1).to(DEVICE)
            
                    return{
                        'facial_motion_style_0': facial_style_feature_0.unsqueeze(0),
                        'facial_motion_style_1': facial_style_feature_1.unsqueeze(0),
                        'body_motion_content': formatted_data['fullbody_feature'].unsqueeze(0),
                    }
            else:
                orientation_feature = self.parent_local_orientation.flatten(1).to(DEVICE)
                facial_style_feature = torch.cat((self.facial_feature_style['jaw'], self.facial_feature_style['face_expr']), dim = -1).to(DEVICE)
                return{
                    'facial_motion_style': facial_style_feature.unsqueeze(0),
                    'body_motion_content': orientation_feature.unsqueeze(0),
                }
            
        else:
            formatted_data = utils.get_formatted_data( 
                                position_feature = self.current_character_local_position,
                                orientation_feature = self.current_character_local_orientation,
                                velocity_feature = self.current_character_local_velocity,
                                )
            return {
                'body_motion_content': formatted_data['fullbody_feature'].unsqueeze(0),
            }




