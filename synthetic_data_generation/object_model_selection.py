from typing import Literal, Tuple, Union, List
from scene_config_manager import SceneConfigManager, SceneConfig
from model_library import MarsPipelineCheckpoint
import object_trajectory_generation as otg
import torch
import os
from collections import namedtuple

ObjectModelsIDsFromOtherScene = namedtuple('ObjectModelsIDsFromOtherScene', ['scene_name', 'scene_checkpoint', 'object_model_ids'])

def add_object_model_to_scene_graph(scene_graph, import_scene_name: str, import_scene_checkpoint: MarsPipelineCheckpoint, object_model_key: str):
    """Add object model to scene."""

    # Update object_model_key with original_scene_name to avoid name conflicts
    object_model_key_number = int(object_model_key)

    original_scene_number = int(import_scene_name)

    object_model_key_with_scene = f"object_{original_scene_number}{str(object_model_key_number).zfill(4)}"

    object_model_config = import_scene_checkpoint.get_object_model_template_config()

    scene_graph.object_models[object_model_key_with_scene] = object_model_config.setup(
        scene_box=scene_graph.scene_box,
        num_train_data=scene_graph.num_train_data,
        object_meta=scene_graph.object_meta,
        obj_feat_dim=scene_graph.config.latent_size if scene_graph.use_object_latent_code else 0,
        car_latents=scene_graph.car_latents,
        car_nerf_state_dict_path=scene_graph.car_nerf_state_dict_path,
    )

    object_model_state = import_scene_checkpoint.get_object_model_state(object_model_key)
    scene_graph.object_models[object_model_key_with_scene].load_state_dict(object_model_state)

    return scene_graph



def get_object_model_ids_from_other_scenes(scene_config_manager: SceneConfigManager, original_scene_config_path: str):
    '''Get object models from other scenes.'''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_scene_config = scene_config_manager.get_scene_config(original_scene_config_path)

    light_condition = original_scene_config.light_condition

    compatible_scene_configs = scene_config_manager.get_scene_configs_filtered(light_condition=light_condition, exclude_scenes=[original_scene_config.scene_name])

    object_models_ids_from_other_scenes = []

    for config in compatible_scene_configs:
        
        # Get config path
        # Load checkpoints and trajectories
        # Check distances and number of frames in which the object is visible
        # Return 'good' object models

        config_path = config.scene_config_path

        print(config_path)
        checkpoint = MarsPipelineCheckpoint(config_path=config_path)

        datamanager_config = checkpoint.config.pipeline.datamanager
        datamanager = datamanager_config.setup(device=device, test_mode='inference')
        train_dataset = datamanager.train_dataset

        scene_cameras = train_dataset.cameras

        obj_location_data = train_dataset.metadata["obj_info"]

        print(f'obj_location_data shape: {obj_location_data.shape}')

        obj_location_data_dyn = obj_location_data.view(
            # len(cameras),
            obj_location_data.shape[0], # == len(cameras) for training images
            # obj_location_data.shape[1],
            datamanager.dataparser.max_input_objects,
            datamanager.dataparser.config.add_input_rows * 3
        )
        print(f'obj_location_data_dyn shape: {obj_location_data_dyn.shape}')


        # Object metadata consistent over all frames/ cameras
        obj_metadata = train_dataset.metadata["obj_metadata"]
        print(f'obj_metadata shape: {obj_metadata.shape}')


        # manipulate_scene_trajectories(initial_cameras, obj_metadata, obj_location_data_dyn)

        # Check distances and filter object models similar to scene_manipulation script
        bounding_box_tracklets = otg.get_bounding_boxes_with_object_ids(batch_objects_dyn=obj_location_data_dyn, obj_metadata=obj_metadata)

        # TODO: Rework this part based on loss function of the object models
        n_closest_objects = 5
        closest_model_ids = otg.get_closest_object_model_ids(bounding_box_tracklets=bounding_box_tracklets, cam2worlds=scene_cameras.cam_to_worlds, n_closest_objects=n_closest_objects)

        print(config.scene_name)

        object_models_ids_from_other_scenes.append(ObjectModelsIDsFromOtherScene(scene_name=config.scene_name, object_model_ids=closest_model_ids, scene_checkpoint=checkpoint))

    return object_models_ids_from_other_scenes


def add_object_models_to_scene_graph(scene_graph, obj_model_ids_from_other_scenes: List[ObjectModelsIDsFromOtherScene]):
    '''Add object models to scene graph.'''

    for obj_model_ids_from_other_scene in obj_model_ids_from_other_scenes:

        import_scene_name = obj_model_ids_from_other_scene.scene_name
        import_scene_checkpoint = obj_model_ids_from_other_scene.scene_checkpoint
        object_model_ids = obj_model_ids_from_other_scene.object_model_ids

        for object_model_key in object_model_ids:
            add_object_model_to_scene_graph(scene_graph=scene_graph, import_scene_name=import_scene_name, import_scene_checkpoint=import_scene_checkpoint, object_model_key=object_model_key)

    return scene_graph

    