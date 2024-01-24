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


class MarsPipelineCheckpoint:
    """Mars pipeline checkpoint."""

    def __init__(self, checkpoint_dir):
        """Initialize."""

        self.checkpoint_dir = checkpoint_dir
        """Directory of run with checkpoint."""

        self.config_path = Path(os.path.join(os.path.dirname(self.checkpoint_dir), '..', "config.yml"))
        """Config path."""

        self._loaded_state = None
        self._state = None
        self._config: TrainerConfig = None

    @property
    def loaded_state(self):
        """Loaded state."""
        if self._loaded_state is None:
            self.load_state()
        return self._loaded_state

    @property
    def state(self):
        """State without module prefixes."""
        if self._state is None:
            self._state = {
                (key[len("module.") :] if key.startswith("module.") else key): value for key, value in self.loaded_state.items()
            }
        return self._state

    @property
    def config(self):
        """Config."""
        if self._config is None:
            self.load_config()
        return self._config

    def load_state(self):
        """Load state from torch checkpoint."""
        self._loaded_state = torch.load(self.checkpoint_dir, map_location="cpu")


    def load_config(self):
        """Load config from yaml file."""
        # Unsafe load, but we have to trust the config file here
        self._config: TrainerConfig = yaml.load(self.config_path.read_text(), Loader=yaml.Loader)

    def get_background_model_config(self):
        """Get background model config."""
        return self.config.pipeline.model.background_model
    
    def get_object_model_template_config(self):
        """Get object model template config."""
        return self.config.pipeline.model.object_model_template
    
    def get_object_model_ids(self):
        """Get object model ids."""
        object_model_ids = set([k.split('.')[2] for k in self.state['pipeline'] if 'object_model' in k])
        return object_model_ids
    
    def get_object_model_keys(self, object_model_id):
        """Get keys for object model."""
        return [k for k in self.state['pipeline'] if object_model_id in k]

    def get_background_model_keys(self):
        return [k for k in self.state['pipeline'] if 'background_model' in k]


@dataclass
class ModelLibrary:
    """Model library."""
    
    checkpoint_dir: str
    """Directory of run with checkpoint."""



    models = []
    """List of models in library?"""

    def load_from_pipeline(self, checkpoint_dir):
        """Load models from pipeline checkpoint."""


        mars_checkpoint = MarsPipelineCheckpoint(checkpoint_dir)

        # config_path = Path(os.path.join(os.path.dirname(checkpoint_dir), '..', "config.yml"))
        # # Unsafe load, but we have to trust the config file here
        # config: TrainerConfig = yaml.load(config_path.read_text(), Loader=yaml.Loader)

        print(mars_checkpoint.config)

        background_model_config = mars_checkpoint.config.pipeline.model.background_model
        object_model_config = mars_checkpoint.config.pipeline.model.object_model_template

        # TODO: Get pipeline datamanager config
        # TODO: Get checkpoint directory
        # TODO: Set datamanager to some datamanager that is registered in nerfstudio
        # TODO: Set pipeline to eval mode
        # TODO: Create pipeline
        # TODO: Create scene graph model
        # TODO: Get metadata from dataparser based on sequence id

        state = mars_checkpoint.state

        pipeline_keys = state['pipeline'].keys()
        background_model_keys = [k for k in state['pipeline'].keys() if 'background_model' in k]
        object_model_keys = [k for k in state['pipeline'].keys() if 'object_model' in k]
        other_pipeline_keys = [k for k in state['pipeline'].keys() if 'object_model' not in k and 'background_model' not in k]

        device_indicator_keys = [k for k in state['pipeline'].keys() if 'device_indicator_param' in k and 'object_model' not in k and 'background_model' not in k]
        pipeline_lpips_keys = [k for k in state['pipeline'].keys() if 'lpips' in k and 'object_model' not in k and 'background_model' not in k]

        object_model_ids = mars_checkpoint.get_object_model_ids()

        background_model_keys = mars_checkpoint.get_background_model_keys()
        single_object_model_keys = mars_checkpoint.get_object_model_keys(next(iter(object_model_ids)))
        print(single_object_model_keys)

        single_object_model_state = {'.'.join(k.split('.')[3:]): v for k, v in state['pipeline'].items() if k in single_object_model_keys}

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

    def main(self):
        """Main function."""
        self.load_from_pipeline(self.checkpoint_dir)


def entrypoint():
    """Entry point."""
    tyro.cli(ModelLibrary).main()


if __name__ == "__main__":
    entrypoint()


