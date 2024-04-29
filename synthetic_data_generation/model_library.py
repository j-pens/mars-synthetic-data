import nerfstudio
import torch
import tyro

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs

from mars.data.mars_pandaset_dataparser import MarsPandasetDataParserConfig, MarsPandasetParser
from mars.models.scene_graph import SceneGraphModelConfig, SceneGraphModel

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

    def __init__(self, config_path):
        """Initialize."""

        self._checkpoint_dir = None
        """Directory of run with checkpoint."""

        self.config_path = config_path
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

    @property
    def checkpoint_dir(self):
        """Checkpoint directory."""
        if self._checkpoint_dir is None:
            self._checkpoint_dir = os.path.join(os.path.dirname(self.config_path), self.config.relative_model_dir)
        return self._checkpoint_dir

    def load_state(self):
        """Load state from torch checkpoint."""
        checkpoint_files_to_steps = {f: int(f[:-len(".ckpt")].split('-')[-1]) for f in os.listdir(self.checkpoint_dir) if f.endswith(".ckpt")}
        checkpoint_files = list(checkpoint_files_to_steps.keys())
        checkpoint_files.sort(key=lambda x: checkpoint_files_to_steps[x])
        self._loaded_state = torch.load(os.path.join(self.checkpoint_dir, checkpoint_files[-1]), map_location="cpu")


    def load_config(self):
        """Load config from yaml file."""
        # Unsafe load, but we have to trust the config file here
        with open(self.config_path, "r") as f:
            self._config = yaml.load(f, Loader=yaml.Loader)

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
        return [k for k in self.state['pipeline'] if str(object_model_id) in k and not 'background_model' in k and 'object_model' in k]

    def get_background_model_keys(self):
        return [k for k in self.state['pipeline'] if 'background_model' in k]

    def get_object_model_state(self, object_model_id):
        """Get state for single object model."""
        object_model_keys = self.get_object_model_keys(object_model_id)
        # print(object_model_keys)
        single_object_model_state = {'.'.join(k.split('.')[3:]): v for k, v in self.state['pipeline'].items() if k in object_model_keys}
        return single_object_model_state

    def get_background_model_state(self):
        """Get background model state."""
        background_model_keys = self.get_background_model_keys()
        background_model_state = {'.'.join(k.split('.')[2:]): v for k, v in self.state['pipeline'].items() if k in background_model_keys}
        return background_model_state

    def get_dataparser_config(self):
        """Returns dataparser config with corrected data path."""
        dataparser_config = self.config.pipeline.datamanager.dataparser
        dataparser_config.data = Path(self.config.data)
        return dataparser_config

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

        background_model_config = mars_checkpoint.config.pipeline.model.background_model
        object_model_config = mars_checkpoint.config.pipeline.model.object_model_template

        # TODO: Get pipeline datamanager config
        # TODO: Get checkpoint directory
        # TODO: Set datamanager to some datamanager that is registered in nerfstudio
        # TODO: Create scene graph model
        # TODO: Get metadata from dataparser based on sequence id

        state = mars_checkpoint.state

        pipeline_keys = state['pipeline'].keys()
        background_model_keys = [k for k in state['pipeline'].keys() if 'background_model' in k]
        object_model_keys = [k for k in state['pipeline'].keys() if 'object_model' in k]
        other_pipeline_keys = [k for k in state['pipeline'].keys() if 'object_model' not in k and 'background_model' not in k]

        device_indicator_keys = [k for k in state['pipeline'].keys() if 'device_indicator_param' in k and 'object_model' not in k and 'background_model' not in k]
        pipeline_lpips_keys = [k for k in state['pipeline'].keys() if 'lpips' in k and 'object_model' not in k and 'background_model' not in k]

        dataparser_config: MarsPandasetDataParserConfig = mars_checkpoint.get_dataparser_config()
        dataparser: MarsPandasetParser = dataparser_config.setup()

        dataparser_outputs: DataparserOutputs = dataparser.get_dataparser_outputs(split="train")

        # print(dataparser_outputs.metadata.keys())
        print(dataparser_outputs.metadata['obj_info'].view(
            dataparser_outputs.metadata['obj_info'].shape[0], 
            dataparser_outputs.metadata['obj_info'].shape[1],
            mars_checkpoint.config.pipeline.model.max_num_obj,
            dataparser.config.add_input_rows * 3
        )[:, :, 0, :])
        # print(dataparser_outputs.metadata['obj_metadata'])
        # print(dataparser_outputs.metadata['scene_obj'])
        obj_model_key = [k for k in dataparser_outputs.metadata['scene_obj']]
        # print(obj_model_key)
        # print((type(obj_model_key[0]), obj_model_key[0]))
        # print(max(dataparser_outputs.metadata['scene_obj']))

        scene_box: SceneBox = dataparser_outputs.scene_box

        num_train_data = mars_checkpoint.config.pipeline.datamanager.dataparser.last_frame - mars_checkpoint.config.pipeline.datamanager.dataparser.first_frame
        assert num_train_data == 40

        # object_model_config = mars_checkpoint.get_object_model_template_config()

        # obj_model = object_model_config.setup(scene_box=scene_box, num_train_data=num_train_data, object_meta=dataparser_outputs.metadata, obj_feat_dim=0)
        # obj_model.load_state_dict(single_object_model_state)


        # bg scale usually 1.0
        # Hard-coded in scene_graph.py
        bg_scale = 1.0
        bg_scene_box = SceneBox(
            aabb=torch.tensor(
                [[-bg_scale, -bg_scale, -bg_scale], [bg_scale, bg_scale, bg_scale]], dtype=torch.float32
            )
        )

        background_model_state = mars_checkpoint.get_background_model_state()

        background_model_config = mars_checkpoint.get_background_model_config()
        background_model = background_model_config.setup(scene_box=bg_scene_box, num_train_data=num_train_data, object_meta=dataparser_outputs.metadata, obj_feat_dim=0)
        background_model.load_state_dict(background_model_state)

        scene_graph_config: SceneGraphModelConfig = mars_checkpoint.config.pipeline.model
        scene_graph_model: SceneGraphModel = scene_graph_config.setup(
            object_meta=dataparser_outputs.metadata,
            scene_box=scene_box,
            num_train_data=num_train_data, 
            scale_factor=dataparser_outputs.metadata['scale_factor'], 
            obj_feat_dim=None, 
            use_car_latents=False, 
            car_latents=None, 
            car_nerf_state_dict_path=None,
            use_depth=False,
            use_semantic=False
        )




        scene_graph_model.background_model = background_model

        for k in scene_graph_model.object_models:
            # print(k)
            state = mars_checkpoint.get_object_model_state(k)
            # print(state.keys())
            scene_graph_model.object_models[k].load_state_dict(state)



    def main(self):
        """Main function."""
        self.load_from_pipeline(self.checkpoint_dir)


def entrypoint():
    """Entry point."""
    tyro.cli(ModelLibrary).main()


if __name__ == "__main__":
    entrypoint()


