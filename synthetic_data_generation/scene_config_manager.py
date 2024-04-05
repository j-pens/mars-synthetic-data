import yaml
import os
import tyro
from typing import Literal, Tuple, Union, List
from dataclasses import dataclass, field
import random

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
            print(f'Loading config storage from {self.config_storage_path}.')
            self.config_storage = self.load_config_storage()
        else:
            print(f'No config storage found at {self.config_storage_path}. Creating new config storage.')
            self.config_storage = {}

    def load_config_storage(self):
        with open(self.config_storage_path, 'r') as file:
            config_storage = yaml.load(file, Loader=yaml.FullLoader)
        return config_storage
    

    def get_scene_config_path(self, scene_name):
        return self.config_storage[scene_name].scene_config_path
    

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

    def config_path_in_config_storage(self, scene_config_path):
        for v in self.config_storage.values():
            for scene_config in v:
                if scene_config.scene_config_path == scene_config_path:
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
    
    def get_scene_configs_filtered(self, match_all=True, **kwargs) -> List[SceneConfig]:
        exclude_scenes = []
        scene_configs = []
            
        if 'exclude_scenes' in kwargs:
            exclude_scenes = kwargs.pop('exclude_scenes')

        match_fn = all if match_all else any
        for v in self.config_storage.values():
            for scene_config in v:
                if match_fn((any(getattr(scene_config, k, None) == item for item in v) if isinstance(v, list) else getattr(scene_config, k, None) == v) for k, v in kwargs.items()) and str(scene_config.scene_name) not in exclude_scenes:
                    scene_configs.append(scene_config)
        return scene_configs


    def get_scene_configs_sampled(self, n_samples: int, **kwargs) -> List[SceneConfig]:
        scene_configs = self.get_scene_configs_filtered(**kwargs)
        return random.sample(scene_configs, n_samples)


    def save_config_storage(self):
        with open(self.config_storage_path, 'w') as file:
            yaml.dump(self.config_storage, file)



def create_scene_config_store(config_store_path: str, scene_config_path: str=''):
    '''Create a scene config storage file and add a scene config to it. 
    If the file already exists, the scene config is added to it.'''

    scene_config_manager = SceneConfigManager(config_store_path)

    if scene_config_path != '':
        scene_config_manager.add_scene_config(scene_config_path)

    scene_config_manager.save_config_storage()



def add_scene_config(config_store_path: str, scene_config_path: str):
    '''Add a scene config to the scene config storage file.'''
    scene_config_manager = SceneConfigManager(config_store_path)

    if scene_config_path != '':
        scene_config_manager.add_scene_config(scene_config_path)

    scene_config_manager.save_config_storage()


def get_day_scene_configs(config_store_path: str):
    '''Get all day scene configs from the scene config storage file.'''
    scene_config_manager = SceneConfigManager(config_store_path)

    day_configs = scene_config_manager.get_scene_configs_filtered(light_condition='day')

    print([config.scene_name for config in day_configs])
    

def main():
    tyro.extras.subcommand_cli_from_dict(
        {
            'new': create_scene_config_store,
            'add': add_scene_config,
            'get_day_configs': get_day_scene_configs
        }
    )




if __name__ == '__main__':
    main()