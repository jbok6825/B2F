import torch
from torch.utils.data import Dataset
import pickle
import os
import process_dataset.utils as utils
from process_dataset.Constant import *
import random
'''
processd_data = {
        'pose_body': motion_parms['pose_body'],
        'pose_hand': motion_parms['pose_hand'],
        'pose_jaw': motion_parms['pose_jaw'], 
        'face_expr': motion_parms['face_expr'][:, :10]
    }
'''
class CustomDataset(Dataset):
    # style dataset의 length는 120(4초)~180(6초)
    # motion dataset의 length는 180(6초)
    def __init__(self, device, path_dataset_dir, only_style = False, same_mode = False, smpl_mode = False):

        self.device = device
        self.path_dataset_dir = path_dataset_dir
        self.only_style = only_style
        self.same_mode = same_mode
        self.smpl_mode = smpl_mode
        if self.only_style == False:
            if self.same_mode == True:
                self._same_feature_list = torch.tensor([], device=self.device)
            elif self.smpl_mode == True:
                self._orientation_feature_list = torch.tensor([], device=self.device)
            else:
                self._position_feature_list = torch.tensor([], device=self.device) #(file_num, 180, 3)
                self._velocity_feature_list = torch.tensor([], device=self.device)  #(file_num, 180, 3)
                self._orientation_feature_list = torch.tensor([], device=self.device)
        self._style_code_list = torch.tensor([], device=self.device)  #(file_num, )
        self._face_expr_feature_list = torch.tensor([], device=self.device)  #(file_num, 180, 50)
        self._jaw_feature_list = torch.tensor([], device=self.device)  #(file_num, 180, 3)

        self._set_feature_list()

    def _set_feature_list(self):
        file_path_list = os.listdir(self.path_dataset_dir)
        file_path_list.sort()
        for file_path in file_path_list:
            with open(self.path_dataset_dir+"/"+file_path, "rb") as file:
                dataset = pickle.load(file)
                if self.only_style == False:
                    if self.same_mode == True:
                        self._same_feature_list = torch.cat((self._same_feature_list, dataset['same_feature'].to(self.device)), dim = 0)
                    elif self.smpl_mode == True:
                        self._orientation_feature_list = torch.cat((self._orientation_feature_list, dataset['orientation'].to(self.device)), dim = 0)
                    else:
                        self._position_feature_list = torch.cat((self._position_feature_list, dataset['position'].to(self.device)), dim = 0)
                        self._orientation_feature_list = torch.cat((self._orientation_feature_list, dataset['orientation'].to(self.device)), dim = 0)
                        self._velocity_feature_list = torch.cat((self._velocity_feature_list, dataset['velocity'].to(self.device)), dim = 0)
                self._face_expr_feature_list = torch.cat((self._face_expr_feature_list, dataset['face_expr'].to(self.device)), dim = 0)
                self._jaw_feature_list = torch.cat((self._jaw_feature_list, dataset['jaw'].to(self.device)), dim = 0)
                self._style_code_list = torch.cat((self._style_code_list, dataset['style_code_list'].to(self.device)), dim = 0)



    def __getitem__(self, index):

        face_expr_style_feature = self._face_expr_feature_list[index]
        jaw_style_feature = self._jaw_feature_list[index]
        style_code = self._style_code_list[index]

        face_expr_feature = self._face_expr_feature_list[index]
        jaw_feature = self._jaw_feature_list[index]
        style_code = self._style_code_list[index]

        if self.only_style == False:
            if self.same_mode == True:
                same_feature = self._same_feature_list[index]
                facial_feature = torch.cat((jaw_feature, face_expr_feature), dim = -1)
                facial_style_feature = torch.cat((jaw_style_feature, face_expr_style_feature), dim = -1)
                formatted_data = {
                                'facial_feature': facial_feature.to(self.device),
                                'fullbody_feature': same_feature.to(self.device),
                                'facial_style_feature':facial_style_feature.to(self.device),
                                'style_code': style_code.to(self.device)}
            elif self.smpl_mode == True:
                orientation_feature = self._orientation_feature_list[index]
            
                facial_feature = torch.cat((jaw_feature, face_expr_feature), dim = -1)
                facial_style_feature = torch.cat((jaw_style_feature, face_expr_style_feature), dim = -1)
                formatted_data = {
                                'facial_feature': facial_feature.to(self.device),
                                'facial_style_feature':facial_style_feature.to(self.device),
                                'style_code': style_code.to(self.device),
                                'fullbody_feature' : orientation_feature.flatten(1).to(self.device)}
                

            else:
                position_feature = self._position_feature_list[index]
                orientation_feature = self._orientation_feature_list[index]
                velocity_feature = self._velocity_feature_list[index]
            



                formatted_data = utils.get_formatted_data( 
                                            position_feature = position_feature,
                                            orientation_feature = orientation_feature,
                                            velocity_feature = velocity_feature,
                                            face_expr_feature = face_expr_feature,
                                            jaw_feature = jaw_feature,
                                            face_expr_style_feature = face_expr_style_feature,
                                            jaw_style_feature = jaw_style_feature,
                                            style_code = style_code)
            
            # n = self._style_code_list.shape[0]
            # random_index = torch.randint(0, n, (1,)).item()
            # random_face_expr_style_feature = self._face_expr_feature_list[random_index]
            # random_jaw_style_feature = self._jaw_feature_list[random_index]

            # random_facial_style_feature = torch.cat((random_jaw_style_feature, random_face_expr_style_feature), dim = -1).to(DEVICE)
            # random_style_code = self._style_code_list[random_index]

            # formatted_data["random_facial_style_feature"] = random_facial_style_feature
            # formatted_data["random_style_code"] = random_style_code.to(DEVICE)
            

        else:
            facial_style_feature = torch.cat((jaw_style_feature, face_expr_style_feature), dim = -1).to(self.device)
            formatted_data = {'facial_style_feature':facial_style_feature,
                              'style_code': style_code.to(self.device)}

        return formatted_data
    
    def sample_style_data(self, style_code):
        indices = (self._style_code_list == style_code).nonzero(as_tuple=True)[0]

        if len(indices) > 0:
            random_index = indices[torch.randint(len(indices), (1,)).item()]
            
            return self.__getitem__(random_index)['facial_style_feature']
        else:
            print("No indices found for the specific value.")

                
    
    def __len__(self):
        return self._style_code_list.shape[0]
    

def collate_fn(batch):
    style_frame_length = 60
    content_frame_length = random.randint(120, 180)

    max_start_idx_style = 180 - style_frame_length
    max_start_idx_content = 180- content_frame_length

    for item in batch:

        start_idx_style = torch.randint(low=0, high=max_start_idx_style+1, size=(1,)).item()
        item['facial_style_feature'] = item['facial_style_feature'][start_idx_style : start_idx_style + style_frame_length, :]

        start_idx_content = torch.randint(low = 0, high=max_start_idx_content+1, size=(1,)).item()
        if 'fullbody_feature' in item and 'facial_feature' in item:
            item['fullbody_feature'] = item['fullbody_feature'][start_idx_content:start_idx_content+content_frame_length]
            item['facial_feature'] = item['facial_feature'][start_idx_content:start_idx_content+content_frame_length]
            

    collated_batch = {key: torch.stack([item[key] for item in batch]) for key in batch[0]}

    return collated_batch


def collate_fn_same_timing(batch):

    frame_length = random.randint(60, 90)

    max_start_idx_content = 180- frame_length

    for item in batch:

        start_idx_content = torch.randint(low = 0, high=max_start_idx_content+1, size=(1,)).item()
        if 'fullbody_feature' in item and 'facial_feature' in item:
            item['fullbody_feature'] = item['fullbody_feature'][start_idx_content:start_idx_content+frame_length]
            item['facial_feature'] = item['facial_feature'][start_idx_content:start_idx_content+frame_length]
            item['facial_style_feature'] = item['facial_style_feature'][start_idx_content:start_idx_content+frame_length]
            

    collated_batch = {key: torch.stack([item[key] for item in batch]) for key in batch[0]}

    return collated_batch


def collate_fn_pairs(batch):
    style_frame_length = random.randint(120, 180)
    content_frame_length = random.randint(120, 180)

    max_start_idx_style = 180- style_frame_length
    max_start_idx_content = 180 - content_frame_length

    for item in batch:
        start_idx_style = torch.randint(low=0, high=max_start_idx_style+1, size=(1,)).item()
        item['facial_style_feature'] = item['facial_style_feature'][start_idx_style : start_idx_style + style_frame_length, :]

        start_idx_content = torch.randint(low=0, high=max_start_idx_content+1, size=(1,)).item()
        if 'fullbody_feature' in item and 'facial_feature' in item:
            item['fullbody_feature'] = item['fullbody_feature'][start_idx_content:start_idx_content+content_frame_length]
            item['facial_feature'] = item['facial_feature'][start_idx_content:start_idx_content+content_frame_length]

    batched_pairs = []
    for i in range(0, len(batch), 2):
        if i+1 < len(batch):
            # 두 개의 item을 결합하여 하나의 딕셔너리로 만듦
            paired_dict = {}
            for key in batch[i]:
                paired_dict[key] = torch.stack([batch[i][key], batch[i+1][key]])
            batched_pairs.append(paired_dict)

    return batched_pairs


def collate_fn_new(batch):
    style_frame_length = random.randint(90, 120)
    content_frame_length = random.randint(120, 180)

    max_start_idx_style = 180 - style_frame_length
    max_start_idx_content = 180- content_frame_length

    for item in batch:

        start_idx_style = torch.randint(low=0, high=max_start_idx_style+1, size=(1,)).item()
        item['facial_style_feature'] = item['facial_style_feature'][start_idx_style : start_idx_style + style_frame_length, :]

        start_idx_content = torch.randint(low = 0, high=max_start_idx_content+1, size=(1,)).item()
        if 'fullbody_feature' in item and 'facial_feature' in item:
            item['fullbody_feature'] = item['fullbody_feature'][start_idx_content:start_idx_content+content_frame_length]
            item['facial_feature'] = item['facial_feature'][start_idx_content:start_idx_content+content_frame_length]
            

    collated_batch = {key: torch.stack([item[key] for item in batch]) for key in batch[0]}

    return collated_batch

def collate_fn_new_fixed_length_content(batch):
    style_frame_length = random.randint(120, 180)
    content_frame_length = 90

    max_start_idx_style = 180 - style_frame_length
    max_start_idx_content = 180- content_frame_length

    for item in batch:

        start_idx_style = torch.randint(low=0, high=max_start_idx_style+1, size=(1,)).item()
        item['facial_style_feature'] = item['facial_style_feature'][start_idx_style : start_idx_style + style_frame_length, :]

        start_idx_content = torch.randint(low = 0, high=max_start_idx_content+1, size=(1,)).item()
        if 'fullbody_feature' in item and 'facial_feature' in item:
            item['fullbody_feature'] = item['fullbody_feature'][start_idx_content:start_idx_content+content_frame_length]
            item['facial_feature'] = item['facial_feature'][start_idx_content:start_idx_content+content_frame_length]
            

    collated_batch = {key: torch.stack([item[key] for item in batch]) for key in batch[0]}

    return collated_batch

def collate_fn_new_content_in_style(batch):
    # Define the minimum length for style and content input (고정된 길이)
    style_frame_length = random.randint(90, 120)
    content_frame_length = random.randint(120, 180)

    # Calculate the maximum starting index for style
    max_start_idx_style = 180 - style_frame_length

    # Generate a fixed random start index for style within the allowed range
    start_idx_style = torch.randint(low=0, high=max_start_idx_style + 1, size=(1,)).item()
    start_idx_content = start_idx_style  # content should include the entire style range

    for item in batch:
        # Apply the fixed start indices and lengths to each item
        item['facial_style_feature'] = item['facial_style_feature'][start_idx_style : start_idx_style + style_frame_length, :]

        if 'fullbody_feature' in item and 'facial_feature' in item:
            item['fullbody_feature'] = item['fullbody_feature'][start_idx_content : start_idx_content + content_frame_length]
            item['facial_feature'] = item['facial_feature'][start_idx_content : start_idx_content + content_frame_length]

    # Collate the batch by stacking each feature across items in the batch
    collated_batch = {key: torch.stack([item[key] for item in batch]) for key in batch[0]}

    return collated_batch



def collate_fn_new_fixed_content_in_style(batch):
    # Define fixed frame lengths
    style_frame_length = random.randint(150, 180)
    content_frame_length = random.randint(60, 120)

    # Calculate the maximum starting index for content
    max_start_idx_content = 180 - content_frame_length

    # Generate a fixed random start index for content
    start_idx_content = torch.randint(low=0, high=max_start_idx_content + 1, size=(1,)).item()

    # Ensure style range starts before or at the same point as content and ends after it
    start_idx_style = max(0, start_idx_content + content_frame_length - style_frame_length)

    for item in batch:
        # Apply the fixed start indices and lengths to each item
        if 'facial_style_feature' in item:
            item['facial_style_feature'] = item['facial_style_feature'][start_idx_style : start_idx_style + style_frame_length, :]
        
        if 'fullbody_feature' in item and 'facial_feature' in item:
            item['fullbody_feature'] = item['fullbody_feature'][start_idx_content : start_idx_content + content_frame_length]
            item['facial_feature'] = item['facial_feature'][start_idx_content : start_idx_content + content_frame_length]

    # Collate the batch by stacking each feature across items in the batch
    collated_batch = {key: torch.stack([item[key] for item in batch]) for key in batch[0]}

    return collated_batch




# def main():
#     DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#     customDataset = MainDataset(DB, device = DEVICE)
    
#     print(len(customDataset))
    
#     data = customDataset[0]
#     print(data['fullbody_feature'].shape)
#     print(data['past_fullbody_feature'].shape)
#     print(data['facial_feature'].shape)
#     print(data['past_facial_feature'].shape)

#     print(customDataset[1]['facial_feature'])
#     print(customDataset[10]['past_facial_feature'][13:26])

# if __name__ == "__main__":
#     main()

