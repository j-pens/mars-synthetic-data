import pandaset as ps
from pandaset import DataSet
from typing import Dict
import os
import pandas as pd
from PIL import Image, ImageDraw
import random
import json
import shutil
import mediapy as media

class PandaSetDataSetSaver():

    def __init__(self, root_path):
        self.root_path = root_path
        self.sequence_savers: Dict[str, PandaSetSequenceSaver] = {}
        os.makedirs(root_path, exist_ok=True)
    

    def add_sequence(self, sequence_name: str):
        saver = PandaSetSequenceSaver(dataset_root_path=self.root_path, sequence_name=sequence_name)
        self.sequence_savers[sequence_name] = saver
        return saver

class PandaSetSequenceSaver():

    _lidar: ps.sequence.Lidar = None
    _camera: Dict[str, ps.sequence.Camera] = {}
    _gps: ps.sequence.GPS = None
    _timestamps: ps.sequence.Timestamps = None
    _cuboids: ps.sequence.Cuboids = None
    _semseg: ps.sequence.SemanticSegmentation = None

    _save_state: Dict[str, int]

    # by default leave out lidar
    def __init__(self, dataset_root_path: str, sequence_name, types_of_data=['meta', 'camera', 'annotations/cuboids', 'lidar']):
        self.root_path = os.path.join(dataset_root_path, sequence_name)
        self.sequence_name = sequence_name
        self._data_paths = self.init_sequence_dirs(types_of_data=types_of_data)
        self._save_state = {type_of_data: 0 for type_of_data in types_of_data}

    def init_sequence_dirs(self, types_of_data=['meta', 'camera', 'annotations/cuboids', 'lidar']):
        data_paths = {type_of_data: os.path.join(self.root_path, type_of_data) for type_of_data in types_of_data}

        for data_path in data_paths.values():
            os.makedirs(data_path, exist_ok=True)

        return data_paths
    
    def add_data(self, data, type_of_data: str, **kwargs):

        if type_of_data == 'camera':
            self.save_image(image=data, **kwargs)

    def get_save_state(self, type_of_data):
        return self._save_state[type_of_data]
    

    def get_data_path(self, type_of_data):
        return self._data_paths[type_of_data]
    
    def update_save_state(self, type_of_data):
        self._save_state[type_of_data] += 1

    def save_image(self, image, camera_name: str = 'front_camera'):

        type_of_data = 'camera'

        image_idx = self.get_save_state(type_of_data)


        dir_path = os.path.join(self.get_data_path(type_of_data), camera_name)
        if not camera_name in self._camera:
            os.makedirs(dir_path, exist_ok=True)
            self._camera[camera_name] = 'TODO:Add cameras object here instead?'

        image_path = os.path.join(dir_path, f'{image_idx:02d}.jpg')

        media.write_image(image_path, image, fmt='jpeg')

        self.update_save_state(type_of_data)


    def save_cuboid_info(self, data: list[pd.DataFrame]):
        
        type_of_data = 'annotations/cuboids'


        for cuboids in data:
            cuboids_idx = self.get_save_state(type_of_data)

            cuboids_path = os.path.join(self.get_data_path(type_of_data), f'{cuboids_idx:02d}.pkl')

            cuboids.to_pickle(cuboids_path)

            self.update_save_state(type_of_data)


    def save_lidar_poses(self, lidar_poses_path: str):
        shutil.copy(lidar_poses_path, self.get_data_path('lidar'), follow_symlinks=True)



    def save_camera_info(self, camera_name, poses: list[dict], intrinsics: dict):
        dir_path = os.path.join(self.get_data_path('camera'), camera_name)
        if not camera_name in self._camera:
            os.makedirs(dir_path, exist_ok=True)

        poses_path = os.path.join(dir_path, 'poses.json')
        intrinsics_path = os.path.join(dir_path, 'intrinsics.json')

        with open(intrinsics_path, 'w') as f:
            json.dump(intrinsics, f)

        with open(poses_path, 'w') as f:
            json.dump(poses, f)

    def save_meta_data(self, data):
        pass


def intrinsics_to_dict(intrinsics):
    return {
        'fx': intrinsics.fx,
        'fy': intrinsics.fy,
        'cx': intrinsics.cx,
        'cy': intrinsics.cy
    }
           
def main():

    root_path = '/Users/jonas/master_thesis_repos/master-thesis-helpers/pandaset_dataset_saving/synthetic_pandaset'

    pandaset_saver = PandaSetDataSetSaver(root_path=root_path)

    sequence_name = '111001'

    sequence_saver = pandaset_saver.add_sequence(sequence_name=sequence_name)

    for i in range(8):
        image = Image.new('RGB', (640, 480), color='red')

        sequence_saver.add_data(data=image, type_of_data='camera', camera_name='front_left_camera')

    intrinsics = ps.sensors.Intrinsics(fx=random.randint(0, 255), fy=random.randint(0, 255), cx=random.randint(0, 255), cy=random.randint(0, 255))

    sequence_saver.save_camera_info('front_left_camera', poses=None, intrinsics=intrinsics_to_dict(intrinsics))

    test_dataset = DataSet(root_path)

    print(test_dataset.sequences())

if __name__ == '__main__':
    main()

            



    

    


