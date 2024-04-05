# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
synthetic_data_render.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import tyro
from rich.console import Console

from typing_extensions import Literal, assert_never

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.utils.data_utils import get_depth_image_from_path
from nerfstudio.model_components.losses import normalized_depth_scale_and_shift
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps, install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import ItersPerSecColumn
from nerfstudio.viewer.server.utils import three_js_perspective_camera_focal_length

from rendering_utils import insert_spherical_metadata_into_file, _render_trajectory_video
from custom_eval_utils import eval_setup as custom_eval_setup
from scene_manipulation import manipulate_scene_trajectories, get_object_models_from_other_scenes
import scene_manipulation as scm
import object_model_selection as oms
import camera_trajectory_generation as ctg
import object_trajectory_generation as otg
import scene_config_manager
import annotation_generation as ag
import pandaset_saver
import synthetic_data_pipeline_config_manager as sdpcm

import random

CONSOLE = Console(width=120)

@dataclass(init=False)
class SyntheticDataRender:
    """Load a synthetic data config, render multiple scenes, and save to a synthetic data set with annotations."""

    # Specifies number of rays per chunk during eval.
    eval_num_rays_per_chunk: Optional[int] = None

    def __init__(self, config: sdpcm.SyntheticDataPipelineConfig, config_directory: Path = Path('/zfs/penshorn/master_thesis/code/mars-synthetic-data/synthetic_data_generation/configs'), video: bool = False):
        self.config = config
        self.video = video
        sdpcm.add_synthetic_data_pipeline_config(config_directory=config_directory, config=config)
        random.seed(config.seed)
        torch.manual_seed(config.seed)

    def main(self) -> None:
        """Main function."""

        scene_conf_manager = scene_config_manager.SceneConfigManager(self.config.scene_configs_path)

        scene_configs = []
        valid_configs = False
        while not valid_configs:
            scene_configs = scene_conf_manager.get_scene_configs_sampled(self.config.n_scenes, light_condition=self.config.light_conditions, method_name='mars-pandaset-nerfacto-object-wise-recon')
            scene_names = [scene_config.scene_name for scene_config in scene_configs]
            # Check if all scene names are unique
            if len(set(scene_names)) == len(scene_names):
                valid_configs = True

        print(scene_configs)

        synthetic_pandaset_annotation_generator = ag.SyntheticPandaSetAnnotationGenerator(dataset_root_path=str(self.config.original_dataset_root))
        synthetic_pandaset_saver = pandaset_saver.PandaSetDataSetSaver(root_path=str(self.config.synthetic_dataset_root))

        for scene_config in scene_configs:
            
            sequence_saver = synthetic_pandaset_saver.add_sequence(sequence_name=scene_config.scene_name)
            static_cuboids, all_cuboids = synthetic_pandaset_annotation_generator.get_static_cuboids(original_sequence_name=scene_config.scene_name)

            lidar_poses_path = synthetic_pandaset_annotation_generator.get_lidar_poses_path(original_sequence_name=scene_config.scene_name)
            sequence_saver.save_lidar_poses(lidar_poses_path=lidar_poses_path)

            _, pipeline, _, _ = eval_setup(
                Path(scene_config.scene_config_path),
                eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
                test_mode="inference",
            )
            
            cameras = pipeline.datamanager.train_dataset.cameras
            camera_trajectory = ctg.get_camera_trajectory(cameras=cameras, jitter=self.config.camera_jitter)

            cameras_pandaset = synthetic_pandaset_annotation_generator.create_camera_poses(cameras=camera_trajectory)
            cam_poses = cameras_pandaset['poses']
            cam_intrinsics = dict(fx=cameras_pandaset['fx'], fy=cameras_pandaset['fy'], cx=cameras_pandaset['cx'], cy=cameras_pandaset['cy'])
            sequence_saver.save_camera_info(camera_name='front_camera', poses=cam_poses, intrinsics=cam_intrinsics)

            obj_location_data = pipeline.datamanager.train_dataset.metadata["obj_info"]

            print(f'obj_location_data shape: {obj_location_data.shape}')

            obj_location_data_dyn = obj_location_data.view(
                # len(cameras),
                obj_location_data.shape[0], # == len(cameras) for training images
                # obj_location_data.shape[1],
                pipeline.model.config.max_num_obj,
                pipeline.model.config.ray_add_input_rows * 3
            )
            print(f'obj_location_data_dyn shape: {obj_location_data_dyn.shape}')


            # Object metadata consistent over all frames/ cameras
            obj_metadata = pipeline.datamanager.train_dataset.metadata["obj_metadata"]
            print(f'obj_metadata shape: {obj_metadata.shape}')

            # Get original tracklets
            tracklets_by_ids = otg.get_bounding_boxes_with_object_ids(batch_objects_dyn=obj_location_data_dyn, obj_metadata=obj_metadata)
            
            # print(f'Original tracklet indices: {[tracklet.original_indices for tracklet in tracklets_by_ids.values()]}')
            tracklets = list(tracklets_by_ids.values())

            for tracklet in tracklets:
                otg.remove_points_object_not_visible(tracklet=tracklet)

            # Create synthetic object trajectories
            positions, yaws = scm.create_synthetic_trajectories(tracklets=tracklets, config=self.config, n_samples=79)


            obj_location_data_out = scm.create_obj_location_data(positions=positions, yaws=yaws)
            
            print(obj_location_data_out[0].shape)
            print(obj_location_data_out[1])

            pipeline.datamanager.train_dataset.metadata["obj_info"] = obj_location_data_out[0]

            pipeline.model.config.max_num_obj = obj_location_data_out[1]

            # print(f'Obj_location data shape: {obj_location_data.shape}')

            print(f'Synthetic obj_location_data shape: {pipeline.datamanager.train_dataset.metadata["obj_info"].shape}')

            # Adjust max_num_obj in config to match the number of objects in the scene
            # Write the trajectory for each object into one row of the obj_location_data tensor
            # TODO: Consider setting obj_model_id to 0 to indicate that the object is not visible in the scenes and adjust sampling

            synthetic_obj_location_data_dyn = pipeline.datamanager.train_dataset.metadata["obj_info"].view(
                # len(cameras),
                obj_location_data.shape[0], # == len(cameras) for training images
                # obj_location_data.shape[1],
                pipeline.model.config.max_num_obj,
                pipeline.model.config.ray_add_input_rows * 3
            )

            synthetic_tracklets_by_id = otg.get_bounding_boxes_with_object_ids(batch_objects_dyn=synthetic_obj_location_data_dyn, obj_metadata=obj_metadata)
            synthetic_tracklets = list(synthetic_tracklets_by_id.values())

            # Save cuboids
            dynamic_cuboids = synthetic_pandaset_annotation_generator.create_dynamic_cuboids(bounding_box_tracklets=synthetic_tracklets)
            merged_cuboids_list_of_dfs = synthetic_pandaset_annotation_generator.merge_static_and_dynamic_cuboids(static_cuboids=static_cuboids, dynamic_cuboids=dynamic_cuboids)
            print(f'Number of cuboids: {len(merged_cuboids_list_of_dfs)}')
            assert len(merged_cuboids_list_of_dfs) != 0
            sequence_saver.save_cuboid_info(data=merged_cuboids_list_of_dfs)

            # angular_bins_tracklets = oms.get_angular_bins_tracklets(tracklets=synthetic_tracklets_by_id.values(), cam2worlds=cameras.camera_to_worlds, n_bins=4096)
            angular_embeddings_tracklets = oms.get_angular_embeddings_tracklets(tracklets=synthetic_tracklets, cam2worlds=cameras.camera_to_worlds)

            # Get objects from other scenes
            models_from_other_scenes = oms.get_object_models_from_other_scenes(scene_config_manager=scene_conf_manager, original_scene_config_path=scene_config.scene_config_path, synthetic_data_pipeline_config=self.config)

            index_to_scene_name = {}
            # angular_bins_from_other_scenes = []
            angular_embeddings_from_other_scenes = []
            for models_from_other_scene in models_from_other_scenes:
                initial_len = len(angular_embeddings_from_other_scenes)
                angular_embedding_tracklets_from_other_scene = oms.get_angular_embeddings_tracklets(tracklets=models_from_other_scene.bounding_box_tracklets, cam2worlds=models_from_other_scene.cam2worlds)
                angular_embeddings_from_other_scenes.extend(angular_embedding_tracklets_from_other_scene)
                len_after_adding = len(angular_embeddings_from_other_scenes)
                index_to_scene_name.update({i: (models_from_other_scene, i - initial_len) for i in range(initial_len, len_after_adding)})

            
            # diff_matrix = oms.get_histogram_difference(angular_bins_tracklets, angular_bins_from_other_scenes)
            diff_matrix = oms.get_mean_diff_matrix(angular_embeddings_tracklets, angular_embeddings_from_other_scenes)

            # Get best fitting object models for each object in the scene
            # Then add them to the scene graph
            scene_graph = pipeline.model

            object_model_keys_with_scene = []
            for diffs_per_trajectory in diff_matrix:
                sorted_indice = diffs_per_trajectory.argsort()

                # Sample models via index here
                # Bias sampling with decreasing probability for higher differences/ higher indices of indices
                # Get object model id and scene based on selected indice
                # Adjust to sample with decreasing probability for higher indices

                select_object_model_weights = self.config.select_object_model_weights

                n_best_indice = sorted_indice[:len(select_object_model_weights)]
                n_indice = len(n_best_indice)
                possible_indices = [i for i in range(n_indice)]
                weights = select_object_model_weights[:n_indice]
                selected_index_position = random.choices(possible_indices, weights=weights, k=1)[0]
                selected_index = sorted_indice[selected_index_position].item()

                selected_scene_model_object, selected_model_index_in_scene = index_to_scene_name[selected_index]

                scene_name = selected_scene_model_object.scene_name
                scene_checkpoint = selected_scene_model_object.scene_checkpoint
                object_model_key = selected_scene_model_object.object_model_ids[selected_model_index_in_scene]

                object_model_key_with_scene = oms.add_object_model_to_scene_graph(scene_graph=scene_graph, import_scene_name=scene_name, import_scene_checkpoint=scene_checkpoint, object_model_key=object_model_key)
                object_model_keys_with_scene.append(object_model_key_with_scene)

            # Write the object model keys to the metadata
            obj_metadata[1:, 0] = torch.tensor(object_model_keys_with_scene)

            FOV = torch.tensor(([30, 26, 22]), dtype=torch.float32)

            render_width = camera_trajectory.image_width[0].item()
            render_height = camera_trajectory.image_height[0].item()
            
            CONSOLE.print(f'Rendering scene {scene_config.scene_name}!') 

            # Render the scene
            image_generator = _render_trajectory_video(
                pipeline,
                cameras=camera_trajectory,
                output_filename=self.config.synthetic_dataset_root / f'{self.config.name}_{scene_config.scene_name}.mp4',
                rendered_output_names=['rgb'],
                rendered_resolution_scaling_factor=1.0,
                seconds=8,
                output_format='generator' if not self.video else 'video',
                camera_type=CameraType.PERSPECTIVE,
                render_width=render_width,
                render_height=render_height
            )

            for image in image_generator:
                sequence_saver.save_image(image=image, camera_name='front_camera')

def run_synthetic_data_render(config: sdpcm.SyntheticDataPipelineConfig, config_directory: Path = Path('/zfs/penshorn/master_thesis/code/mars-synthetic-data/synthetic_data_generation/configs'), video: bool = False) -> None:
    """Run the synthetic data rendering pipeline."""
    SyntheticDataRender(config=config, config_directory=config_directory, video=video).main()


def run_synthetic_data_render_from_config_path(config_path: Path, video: bool = False) -> None:
    """Run the synthetic data rendering pipeline from a config path."""
    config = sdpcm.load_synthetic_data_pipeline_config_from_path(config_path)
    config_directory = config_path.parent
    run_synthetic_data_render(config=config, config_directory=config_directory, video=video)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    # tyro.cli(run_synthetic_data_render, prog_name="Synthetic Data Rendering Pipeline", description="Render synthetic data scenes and save to a synthetic data set with annotations.")

    tyro.extras.subcommand_cli_from_dict(
        {
            'new': run_synthetic_data_render,
            'from_config': run_synthetic_data_render_from_config_path
        }
    )


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(SyntheticDataRender)  # noqa
