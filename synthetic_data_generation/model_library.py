import nerfstudio
import torch
import tyro

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig

from dataclasses import dataclass, field
import os
import yaml
from pathlib import Path
# TODO: Create pipeline
# TODO: Create scene graph model



# TODO: Load single object model
# TODO: Load background model



@dataclass
class ModelLibrary:
    """Model library."""
    
    checkpoint_path: str
    """Path to checkpoint."""



    models = []
    """List of models in library?"""

    def load_from_pipeline(self, checkpoint_path):
        """Load models from pipeline checkpoint."""
        loaded_state = torch.load(checkpoint_path, map_location="cpu")

        print(loaded_state.keys())

        config_path = Path(os.path.join(os.path.dirname(checkpoint_path), '..', "config.yml"))
        # Unsafe load, but we have to trust the config file here
        config: TrainerConfig = yaml.load(config_path.read_text(), Loader=yaml.Loader)

        print(config)

        background_model_config = config.pipeline.model.background_model
        object_model_config = config.pipeline.model.object_model_template

        # TODO: Get pipeline datamanager config
        # TODO: Get checkpoint directory
        # TODO: Set datamanager to some datamanager that is registered in nerfstudio
        # TODO: Set pipeline to eval mode
        # TODO: Create pipeline
        # TODO: Create scene graph model
        # TODO: Get metadata from dataparser based on sequence id

        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }

        pipeline_keys = state['pipeline'].keys()
        background_model_keys = [k for k in state['pipeline'].keys() if 'background_model' in k]
        object_model_keys = [k for k in state['pipeline'].keys() if 'object_model' in k]
        other_pipeline_keys = [k for k in state['pipeline'].keys() if 'object_model' not in k and 'background_model' not in k]
        # print(other_pipeline_keys)

        device_indicator_keys = [k for k in state['pipeline'].keys() if 'device_indicator_param' in k and 'object_model' not in k and 'background_model' not in k]
        # print(device_indicator_keys)
        pipeline_lpips_keys = [k for k in state['pipeline'].keys() if 'lpips' in k and 'object_model' not in k and 'background_model' not in k]


        n_other_pipeline_keys = len(other_pipeline_keys)
        n_lpips_pipeline_keys = len(pipeline_lpips_keys)
        n_device_indicator_keys = len(device_indicator_keys)
        n_obj_model_keys = len(object_model_keys)
        n_background_model_keys = len(background_model_keys)
        n_pipeline_keys = len(pipeline_keys)

        # print(f"Number of pipeline keys: {n_pipeline_keys} = {n_obj_model_keys} object model keys + {n_background_model_keys} background model keys + {n_lpips_pipeline_keys} lpips pipeline keys + {n_device_indicator_keys} pipeline device indicator keys?")

        object_model_ids = self.get_object_model_ids(pipeline_keys)

        non_model_keys = [k for k in state['pipeline'].keys() if not k.startswith('_model.')]

        background_model_keys = self.get_background_model_keys(pipeline_keys)
        single_object_model_keys = self.get_object_model_keys(pipeline_keys, next(iter(object_model_ids)))
        print(single_object_model_keys)

        single_object_model_state = {'.'.join(k.split('.')[3:]): v for k, v in state['pipeline'].items() if k in single_object_model_keys}
        # print(single_object_model_state.keys())

        # object scene box might vary
        aabb_scale = 1.0
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )
        num_train_data = 40

        object_model_config = NerfactoModelConfig(far_plane=150.0, background_color='black')

        obj_model = object_model_config.setup(scene_box=scene_box, num_train_data=num_train_data)
        obj_model.load_state_dict(single_object_model_state)

        # bg scale usually 1.0
        bg_scale = 1.0
        bg_scene_box = SceneBox(
            aabb=torch.tensor(
                [[-bg_scale, -bg_scale, -bg_scale], [bg_scale, bg_scale, bg_scale]], dtype=torch.float32
            )
        )

        background_model_state = {'.'.join(k.split('.')[2:]): v for k, v in state['pipeline'].items() if k in background_model_keys}

        background_model_config = NerfactoModelConfig(far_plane=150.0, background_color='black')
        background_model = background_model_config.setup(scene_box=bg_scene_box, num_train_data=num_train_data)
        background_model.load_state_dict(background_model_state)

        

    def get_object_model_ids(self, pipeline_keys):
        """Get object model ids."""
        object_model_ids = set([k.split('.')[2] for k in pipeline_keys if 'object_model' in k])
        return object_model_ids
    
    def get_object_model_keys(self, pipeline_keys, object_model_id):
        """Get keys for object model."""
        return [k for k in pipeline_keys if object_model_id in k]

    def get_background_model_keys(self, pipeline_keys):
        return [k for k in pipeline_keys if 'background_model' in k]

    def main(self):
        """Main function."""
        self.load_from_pipeline(self.checkpoint_path)


def entrypoint():
    """Entry point."""
    tyro.cli(ModelLibrary).main()


if __name__ == "__main__":
    entrypoint()


