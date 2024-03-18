import re
from nerfstudio.cameras.cameras import Cameras
from object_trajectory_generation import BoundingBoxTracklet
import object_trajectory_generation as otg

import scene_config_manager as scm

import object_model_selection as oms

from torch_cubic_spline_grids import CubicCatmullRomGrid1d

import torch

def manipulate_scene_trajectories(cameras: Cameras, obj_metadata, obj_location_data):
    """Manipulate scene trajectories."""
    
    # get camera trajectory
    cam2world = cameras.camera_to_worlds
    times = cameras.times

    n_cams = len(cameras)
    
    # TODO: Create trajectories based on camera times
    # --> How to handle multiple cameras, i.e. front, back, ...?
    # TODO: Manipulate camera trajectories
    # TODO: Manipulate object trajectories

    # TODO: 1. Create notebook for testing torch-cubic-spline-grids package


    # TODO: Manipulate the object positions, rotations, dimensions and so on using the bounding box tracklet class instances
    # TODO: Replace obj_metadata with the updated values
    # TODO: Replace obj_location_data_dyn with the updated values
    # TODO: Adjust further pipeline steps if necessary

    # TODO: Return cameras, obj_metadata, obj_location_data_dyn


    bounding_box_tracklets = otg.get_bounding_boxes_with_object_ids(batch_objects_dyn=obj_location_data, obj_metadata=obj_metadata)

    bounding_box_tracklet_keys = list(bounding_box_tracklets.keys())

    print(bounding_box_tracklet_keys)

    bounding_box_tracklets_list = list(bounding_box_tracklets.values())

    for tracklet in bounding_box_tracklets_list:
        tracklet.save(f'pandaset_tracklets/seq_011_reworked_dataparser_001/bounding_box_tracklet_{tracklet.obj_id}.pt')

        otg.remove_points_object_not_visible(tracklet=tracklet)
        otg.remove_physically_implausible_points(tracklet=tracklet)

        

        # tracklet.x += 0.5e-2 # 0.5m

        # random_index = torch.randint(0, len(bounding_box_tracklets_list), (1,)).item()

        # tracklet.obj_model_id = bounding_box_tracklet_keys[random_index]

    # index = 0

    # keys_to_dists = {key: get_min_camera_distance(bounding_box_tracklets[key], cam2worlds=cam2world) for key in bounding_box_tracklet_keys}

    # print(keys_to_dists)

    # sorted_keys_asc_dist = sorted(bounding_box_tracklet_keys, key=lambda x: keys_to_dists[x])

    # print(sorted_keys_asc_dist)

    # print(sorted_keys_asc_dist[index])

    # n_closest_objects = 2

    # random_indices = torch.randint(0, n_closest_objects, (len(bounding_box_tracklet_keys),))
    # random_obj_model_ids = torch.tensor([sorted_keys_asc_dist[i] for i in random_indices])

    # obj_metadata[1:, 0] = random_obj_model_ids

    # exit()

    randomize_object_models(bounding_box_tracklets, obj_metadata, cam2worlds=cam2world, n_closest_objects=5)

    positions, yaws = insert_synthetic_trajectories(bounding_box_tracklets_list, n_samples=n_cams)

    write_to_obj_location_data(obj_location_data, positions, yaws)

    closest_model_ids = otg.get_closest_object_model_ids(bounding_box_tracklets, cam2worlds=cam2world, n_closest_objects=5)

    indices = get_indices_from_object_model_ids(obj_metadata, closest_model_ids) - 1

    obj_location_data = obj_location_data[indices]

    # exit()



def randomize_object_models(bounding_box_tracklets, obj_metadata, cam2worlds, n_closest_objects=5):
    """Randomize object models based on the n_closest_objects object models based on the minimum distance to the camera at any point in the sequence."""

    sorted_keys_asc_dist = otg.get_closest_object_model_ids(bounding_box_tracklets, cam2worlds=cam2worlds, n_closest_objects=n_closest_objects)
    random_indices = torch.randint(0, n_closest_objects, (len(bounding_box_tracklets),))
    random_obj_model_ids = torch.tensor([sorted_keys_asc_dist[i] for i in random_indices])

    obj_metadata[1:, 0] = random_obj_model_ids

    return obj_metadata


def randomize_object_models_given_key_strings(obj_model_ids, obj_metadata):
    random_indices = torch.randint(0, len(obj_model_ids), (obj_metadata.shape[0]-1,))
    random_obj_model_ids = torch.tensor([obj_model_ids[i] for i in random_indices])

    obj_metadata[1:, 0] = random_obj_model_ids

    return obj_metadata


def get_indices_from_object_model_ids(obj_metadata, object_model_ids):
    """Get the indices of the object_model_ids in the obj_metadata tensor."""

    indices = torch.tensor([i for i, obj_model_id in enumerate(obj_metadata[1:, 0]) if obj_model_id in object_model_ids])

    # print(indices)

    return indices


def insert_synthetic_trajectories(tracklets: list[BoundingBoxTracklet], n_samples=79):

    parametrizations = [otg.get_parametrization(tracklet, optimization_steps=5000, add_noise=False, noise_level=0.2, spline_grid_class=CubicCatmullRomGrid1d, print_loss=False, with_optimizer=False) for tracklet in tracklets]

    samples = [otg.sample_with_jitter(n_samples, lower=0, upper=1.0, jitter=0.25) for tracklet in tracklets]

    results = [parametrization(sample.unsqueeze(1)).squeeze().detach() for sample, parametrization in zip(samples, parametrizations)]

    yaws = [otg.calculate_yaw(result) for result in results]

    return results, yaws


def write_to_obj_location_data(obj_location_data, positions, yaws):
    """Write the results to the obj_location_data tensor."""

    for i in range(len(positions)):
        batch_objects_dyn_row = obj_location_data[..., i, :]
    
        # position of the object: x, y, z
        # pos = batch_objects_dyn_row[..., :3]
        # yaw = batch_objects_dyn_row[..., 3]
        batch_objects_dyn_row[..., :3] = positions[i]
        batch_objects_dyn_row[..., 3] = yaws[i]



def get_object_models_from_other_scenes(config_path):
    """Get object models from other scenes."""
    
    scene_config_manager = scm.SceneConfigManager('synthetic_data_generation/scene_configs_decent_miraculix.yaml')

    object_model_ids_from_other_scenes = oms.get_object_model_ids_from_other_scenes(scene_config_manager, config_path)

    return object_model_ids_from_other_scenes