from operator import is_
from networkx import is_negatively_weighted
import yaml
import os
import tyro
from dataclasses import dataclass, field

@dataclass
class SceneConfig(yaml.YAMLObject):

    yaml_tag = u'!SceneConfig'

    scene_name: str
    '''Name of the scene.'''

    scene_config_path: str
    '''Path to the scene configuration file.'''

    light_condition: str
    '''Light condition of the scene.'''

    road_type: str
    '''Type of the road in the scene.'''

    method_name: str
    '''Name of the method used to generate the scene.'''

    cameras: list
    '''List of cameras used to generate the scene.'''

night_sequences = [ '057', '058', '059','063', '064', '065', '066', '067', '068', '069', '070', '071', '072', '073',
                       '074', '077', '078', '079', '149']
lighting_sequences = ['024','084', '085', '086', '088', '089', '090', '091', '092', '093', '094']



class SceneConfigManager():

    def __init__(self, config_store_path):
        self.config_storage_path = config_store_path

        if os.path.exists(self.config_storage_path):
            self.config_storage = self.load_config_storage()
        else:
            self.config_storage = {}

    def load_config_storage(self):
        with open(self.config_storage_path, 'r') as file:
            config_storage = yaml.load(file, Loader=yaml.FullLoader)
        return config_storage
    

    def get_scene_config_path(self, scene_name):
        return self.config_storage[scene_name]
    

    def add_scene_config(self, scene_config_path):
        if self.config_path_in_config_storage(scene_config_path):
            print(f'Scene config {scene_config_path} already in config storage.')
            return
        scene_config = self.get_scene_config(scene_config_path)
        scene_name = scene_config.scene_name
        if scene_name in self.config_storage:
            self.config_storage[scene_name].append(scene_config)
        else: 
            self.config_storage[scene_name] = [scene_config]
        self.save_config_storage()

    def config_path_in_config_storage(self, scene_name_config_path):
        for v in self.config_storage.values():
            for scene_config in v:
                if scene_config.scene_config_path == scene_name_config_path:
                    return True
        return False

    def get_scene_config(self, scene_config_path):
        with open(scene_config_path, 'r') as file:
            full_scene_config = yaml.load(file, Loader=yaml.BaseLoader)

        dataparser = full_scene_config['pipeline']['datamanager']['dataparser']
        scene_name = str(dataparser['seq_name'])
        cameras = dataparser['cameras_name_list']

        is_night = scene_name in night_sequences
        is_frontal_lighting = scene_name in lighting_sequences
        
        if is_night:
            light_condition = 'night'
        elif is_frontal_lighting:
            light_condition = 'frontal_lighting'
        else:
            light_condition = 'day'

        scene_config = SceneConfig(
            scene_name=scene_name,
            scene_config_path=scene_config_path,
            light_condition=light_condition,
            road_type='Not yet implemented.',
            method_name=full_scene_config['method_name'],
            cameras=cameras
        )
        return scene_config
    
    def save_config_storage(self):
        with open(self.config_storage_path, 'w') as file:
            yaml.dump(self.config_storage, file)



def create_scene_config_manager(config_store_path: str, scene_config_path: str=''):
    scene_config_manager = SceneConfigManager(config_store_path)

    if scene_config_path is not None:
        scene_config_manager.add_scene_config(scene_config_path)

    scene_config_manager.save_config_storage()

if __name__ == '__main__':
    tyro.cli(create_scene_config_manager)