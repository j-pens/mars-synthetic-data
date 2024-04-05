"""
annos_test.py
"""
from __future__ import annotations

import json
import os
import stat
import struct
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import mediapy as media
import numpy as np
import pandaset
import torch
import tyro
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from typing_extensions import Literal, assert_never

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps, install_checks
from nerfstudio.utils.eval_utils import eval_setup

from custom_eval_utils import eval_setup as custom_eval_setup
from scene_manipulation import manipulate_scene_trajectories, get_object_models_from_other_scenes
import scene_manipulation as scm
import object_model_selection as oms
import object_trajectory_generation as otg
import annotation_generation as ag

import pandaset_saver


def create_synthetic_annotations(pipeline: Pipeline):

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

    bounding_box_tracklets = otg.get_bounding_boxes_with_object_ids(batch_objects_dyn=obj_location_data_dyn, obj_metadata=obj_metadata)

    dataset_root_path = '/zfs/penshorn/master_thesis/datasets/raw/PandaSet'
    synthetic_pandaset_annotation_generator = ag.SyntheticPandaSetAnnotationGenerator(dataset_root_path=dataset_root_path)

    sequence_name = pipeline.datamanager.dataparser.seq_name
    static_cuboids, all_cuboids = synthetic_pandaset_annotation_generator.get_static_cuboids(original_sequence_name=sequence_name)
    dynamic_cuboids = synthetic_pandaset_annotation_generator.create_dynamic_cuboids(bounding_box_tracklets=list(bounding_box_tracklets.values()))

    merged_cuboids_list_of_dfs = synthetic_pandaset_annotation_generator.merge_static_and_dynamic_cuboids(static_cuboids=static_cuboids, dynamic_cuboids=dynamic_cuboids)

    synthetic_dataset_root = '/zfs/penshorn/master_thesis/datasets/synthetic/PandaSet_000'
    synthetic_pandaset_saver = pandaset_saver.PandaSetDataSetSaver(root_path=synthetic_dataset_root)
    sequence_saver = synthetic_pandaset_saver.add_sequence(sequence_name=sequence_name)
    sequence_saver.save_cuboid_info(data=merged_cuboids_list_of_dfs)

    cameras = pipeline.datamanager.train_dataset.cameras

    cameras_pandaset = synthetic_pandaset_annotation_generator.create_camera_poses(cameras=cameras)
    
    poses = cameras_pandaset['poses']
    intrinsics = dict(fx=cameras_pandaset['fx'], fy=cameras_pandaset['fy'], cx=cameras_pandaset['cx'], cy=cameras_pandaset['cy'])
    sequence_saver.save_camera_info(camera_name='front_camera', poses=poses, intrinsics=intrinsics)

    lidar_path = synthetic_pandaset_annotation_generator.get_lidar_poses_path(original_sequence_name=sequence_name)
    sequence_saver.save_lidar_info(lidar_path=lidar_path)


@dataclass
class RenderTrajectory:
    """Load a checkpoint, render a trajectory, and save to a video file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the renderer outputs to use. rgb, depth, semantics etc. concatenates them along y axis
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb", "objects_rgb", "background"])
    #  Trajectory to render.
    traj: Literal["spiral", "filename"] = "spiral"
    # Scaling factor to apply to the camera image resolution.
    downscale_factor: int = 1
    # Filename of the camera path to render.
    camera_path_filename: Path = Path("camera_path.json")
    # Name of the output file.
    output_path: Path = Path("renders/output.mp4")
    # How long the video should be.
    seconds: float = 8
    # How to save output data.
    output_format: Literal["images", "video"] = "video"
    # Specifies number of rays per chunk during eval.
    eval_num_rays_per_chunk: Optional[int] = None

    def main(self) -> None:
        """Main function."""

        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
        )

        create_synthetic_annotations(pipeline)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderTrajectory).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RenderTrajectory)  # noqa