from process_dataset.Constant import *
import os

class FileManager:
    def __init__(self):
        self.sub_dir_list = []
        self.sub_dir_folder_list = []
        self.file_path_list = []
        self.style_file_path_list = []
        self.bvh_file_path_list = []

        self.set_sub_dir_info()
        self.set_file_path()

    def set_sub_dir_info(self):
        self.sub_dir_list = os.listdir(PATH_DB_ORIGIN)
        self.sub_dir_list.sort()
        
        for dir_name in self.sub_dir_list:
            sub_dir_folder = os.listdir(PATH_DB_ORIGIN + "/"+ dir_name)
            sub_dir_folder.sort()
            self.sub_dir_folder_list.append(sub_dir_folder)

    def set_file_path(self):
        for index_dir in range(len(self.sub_dir_list)):
            dir_name = self.sub_dir_list[index_dir]
            for folder_name in self.sub_dir_folder_list[index_dir]:
                file_path = os.listdir(PATH_DB_ORIGIN + "/" + dir_name + "/" + folder_name)
                file_path.sort()
                for file in file_path:
                    self.file_path_list.append(PATH_DB_ORIGIN + "/" + dir_name + "/" + folder_name + "/" + file)
                    self.style_file_path_list.append(PATH_DB_STYLE_ORIGIN + "/" + dir_name + "/" + folder_name + "/" + file[:-3]+"txt")
                    self.bvh_file_path_list.append(PATH_DB_BVH_ORIGIN+ "/" + dir_name + "/" + folder_name + "/" + file[:-3]+"bvh")
    
def get_style_path_from_file_path(file_path):
    file_name = file_path.split("/")[-1]
    folder_name = file_path.split("/")[-2]
    dir_name = file_path.split("/")[-3]

    style_path = PATH_DB_STYLE_ORIGIN + "/"+dir_name + "/"+ folder_name +"/"+ file_name[:-3]+"txt"

    return style_path

def get_bvh_path_from_file_path(file_path):
    file_name = file_path.split("/")[-1]
    folder_name = file_path.split("/")[-2]
    dir_name = file_path.split("/")[-3]

    bvh_path = PATH_DB_BVH_ORIGIN + "/"+dir_name+"/"+folder_name+"/"+file_name[:-3]+"bvh"

    return bvh_path





fileManager = FileManager()