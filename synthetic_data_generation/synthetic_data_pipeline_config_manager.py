from ast import Dict
from pathlib import Path
import yaml
import os
import tyro
from typing import Literal, Tuple, Union, List, Dict
from dataclasses import dataclass, field
import random

@dataclass
class SyntheticDataPipelineConfig(yaml.YAMLObject):

    yaml_tag = u'!SyntheticDataPipelineConfig'

    synthetic_dataset_root: Path
    '''Root directory of the synthetic dataset.'''

    name: str
    '''Name of the configuration.'''

    n_scenes: int = 1
    '''Number of scenes to generate.'''

    light_conditions: List[Literal['day', 'night', 'frontal_lighting']] = field(default_factory=lambda: ['day'])
    '''List of light conditions to generate the scenes for.'''

    scene_configs_path: Path = Path('/zfs/penshorn/master_thesis/code/mars-synthetic-data/synthetic_data_generation/scene_configs_decent_miraculix.yaml')
    '''Path to the scene configuration file.'''

    original_dataset_root: Path = Path('/zfs/penshorn/master_thesis/datasets/raw/PandaSet')
    '''Root directory of the original dataset.'''

    trajectory_sampling_jitter: float = 0.25
    '''Jitter for sampling the synthetic trajectories.'''

    spline_optimization_steps: int = 5000
    '''Number of optimization steps for the spline optimization.'''

    spline_add_noise_to_observations: bool = False
    '''Add noise to the observations during spline optimization.'''

    spline_max_noise: float = 0.2
    '''Maximum noise level for the spline optimization.'''

    spline_grid_class: str = 'catmull_rom'
    '''Class of the spline grid used for the spline optimization.'''

    print_spline_loss: bool = False
    '''Print the loss during spline optimization.'''

    return_spline_optimizer: bool = False
    '''Return the spline optimizer during spline optimization.'''

    spline_max_control_points: int = 10
    '''Maximum number of control points for the spline optimization.'''

    max_acceleration_check: float = 5
    '''Maximum acceleration for the spline optimization.'''

    max_velocity_check: float = 50
    '''Maximum velocity for the spline optimization.'''

    n_closest_objects: int = 5
    '''Number of closest objects to consider for object model selection.'''

    n_frames_min_objects: int = 15
    '''Minimum number of frames for object model selection.'''

    max_object_distance: float = 25
    '''Maximum distance of objects to consider for object model selection.'''

    camera_jitter: float = 0.0005
    '''Jitter for the camera positions.'''

    select_object_model_weights: List[int] = field(default_factory=lambda: [80, 15, 5])
    '''Weights for the object model selection. Length must be equal to n_best_object_models.'''

    seed: int = int.from_bytes(os.urandom(4), 'big')
    '''Seed for the random number generator.'''


class SyntheticDataPipelineConfigManager():

    def __init__(self, config_directory: Path):
        self.config_directory = config_directory

        if os.path.exists(self.config_directory):
            print(f'Loading configs from {self.config_directory}.')
        else:
            print(f'No config directory found at {self.config_directory}. Creating new directory.')
            os.makedirs(self.config_directory)
        self.config_paths = self.get_config_paths()

    def get_config_paths(self):
        return [os.path.join(self.config_directory, file) for file in os.listdir(self.config_directory) if file.endswith('.yaml')]
    
    def create_synthetic_data_pipeline_config(self, **kwargs):
        config = SyntheticDataPipelineConfig(**kwargs)
        return config
    
    def save_synthetic_data_pipeline_config(self, config: SyntheticDataPipelineConfig, config_name: str):
        with open(os.path.join(self.config_directory, f'{config_name}.yaml'), 'w') as file:
            yaml.dump(config, file)

    def load_synthetic_data_pipeline_config(self, config_name: str):
        with open(os.path.join(self.config_directory, f'{config_name}.yaml'), 'r') as file:
            config = yaml.load(file, Loader=yaml.Loader)
        return config
    
    def add_synthetic_data_pipeline_config(self, config_name: str, config: SyntheticDataPipelineConfig):
        self.save_synthetic_data_pipeline_config(config, config_name)
        self.config_paths = self.get_config_paths()

    def get_synthetic_data_pipeline_config(self, config_name: str):
        return self.load_synthetic_data_pipeline_config(config_name)
    

def add_synthetic_data_pipeline_config(config_directory: Path, config: SyntheticDataPipelineConfig) -> None:
    config_manager = SyntheticDataPipelineConfigManager(config_directory)
    config_manager.add_synthetic_data_pipeline_config(config.name, config=config)


def load_synthetic_data_pipeline_config_from_path(config_path: Path):
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)
    return config


def main():
    # tyro.extras.subcommand_cli_from_dict(
    #     {
    #         'add': add_synthetic_data_pipeline_config
    #     }
    # )
    tyro.cli(add_synthetic_data_pipeline_config)




if __name__ == '__main__':
    main()