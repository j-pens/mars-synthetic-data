from nerfstudio.cameras.cameras import Cameras
from object_trajectory_generation import BoundingBoxTracklet
import object_trajectory_generation as otg

import scene_config_manager as scm

import object_model_selection as oms
import synthetic_data_pipeline_config_manager as sdpcm

from torch_cubic_spline_grids import CubicCatmullRomGrid1d

import torch

import random

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

    bounding_box_tracklets = otg.get_bounding_boxes_with_object_ids(
        batch_objects_dyn=obj_location_data, obj_metadata=obj_metadata)

    bounding_box_tracklet_keys = list(bounding_box_tracklets.keys())

    print(bounding_box_tracklet_keys)

    bounding_box_tracklets_list = list(bounding_box_tracklets.values())

    for tracklet in bounding_box_tracklets_list:
        tracklet.save(
            f'pandaset_tracklets/seq_011_reworked_dataparser_001/bounding_box_tracklet_{tracklet.obj_id}.pt')

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

    randomize_object_models(
        bounding_box_tracklets, obj_metadata, cam2worlds=cam2world, n_closest_objects=5)

    positions, yaws = create_synthetic_trajectories(
        bounding_box_tracklets_list, n_samples=n_cams)

    write_to_obj_location_data(obj_location_data, positions, yaws)

    closest_model_ids = otg.get_closest_object_model_ids(
        bounding_box_tracklets, cam2worlds=cam2world, n_closest_objects=5)

    indices = get_indices_from_object_model_ids(
        obj_metadata, closest_model_ids) - 1

    obj_location_data = obj_location_data[indices]

    # exit()


def randomize_object_models(bounding_box_tracklets, obj_metadata, cam2worlds, n_closest_objects=5):
    """Randomize object models based on the n_closest_objects object models based on the minimum distance to the camera at any point in the sequence."""

    sorted_keys_asc_dist = otg.get_closest_object_model_ids(
        bounding_box_tracklets, cam2worlds=cam2worlds, n_closest_objects=n_closest_objects)
    random_indices = torch.randint(
        0, n_closest_objects, (len(bounding_box_tracklets),))
    random_obj_model_ids = torch.tensor(
        [sorted_keys_asc_dist[i] for i in random_indices])

    obj_metadata[1:, 0] = random_obj_model_ids

    return obj_metadata


def randomize_object_models_given_key_strings(obj_model_ids, obj_metadata):
    random_indices = torch.randint(
        0, len(obj_model_ids), (obj_metadata.shape[0]-1,))
    random_obj_model_ids = torch.tensor(
        [obj_model_ids[i] for i in random_indices])

    obj_metadata[1:, 0] = random_obj_model_ids

    return obj_metadata


def get_indices_from_object_model_ids(obj_metadata, object_model_ids):
    """Get the indices of the object_model_ids in the obj_metadata tensor."""

    indices = torch.tensor([i for i, obj_model_id in enumerate(
        obj_metadata[1:, 0]) if obj_model_id in object_model_ids])

    # print(indices)

    return indices


def create_synthetic_trajectories(tracklets: list[BoundingBoxTracklet], config: sdpcm.SyntheticDataPipelineConfig, n_samples=79):

    parametrizations = [otg.get_parametrization_2d(tracklet, optimization_steps=config.spline_optimization_steps, add_noise=config.spline_add_noise_to_observations, noise_level=config.spline_max_noise,
                                                spline_grid_class=CubicCatmullRomGrid1d, print_loss=config.print_spline_loss, with_optimizer=config.return_spline_optimizer) if len(tracklet.x) > 1 else None for tracklet in tracklets]
    
    new_indices_samples = [otg.sample_with_jitter_from_indices(tracklet.original_indices, jitter=0) if len(tracklet.x) > 1 else tracklet.original_indices for tracklet in tracklets]

    results = [parametrization(index_sample[1].unsqueeze(1)).squeeze().detach(
    ) if parametrization is not None and len(index_sample) > 1 else None for index_sample, parametrization in zip(new_indices_samples, parametrizations)]

    yaws = [otg.calculate_yaw(result) if result is not None and len(tracklet.x) > 1 else tracklet.yaw for result, tracklet in zip(results, tracklets)]

    indices = [index_sample[0] if len(index_sample) > 0 else index_sample for index_sample in new_indices_samples]

    results_original_height = [torch.cat((result, tracklet.z.unsqueeze(-1)), dim=-1) if result is not None and len(tracklet.x) > 1 else torch.cat((tracklet.x.unsqueeze(-1), tracklet.y.unsqueeze(-1), tracklet.z.unsqueeze(-1)), dim=-1) for result, tracklet in zip(results, tracklets)]

    return indices, results_original_height, yaws


def write_to_obj_location_data(obj_location_data, positions, yaws):
    """Write the results to the obj_location_data tensor."""

    print(len(positions))

    for i in range(len(positions)):
        batch_objects_dyn_row = obj_location_data[..., i, :]

        # position of the object: x, y, z
        # pos = batch_objects_dyn_row[..., :3]
        # yaw = batch_objects_dyn_row[..., 3]
        batch_objects_dyn_row[..., :3] = positions[i]
        batch_objects_dyn_row[..., 3] = yaws[i]


def create_obj_location_data(indice, positions, yaws):
    """Create the obj_location_data tensor from the positions and yaws."""

    assert len(positions) == len(yaws)

    print(indice)

    n_objects = len(positions)

    padded_positions = []
    padded_yaws = []
    for idx, position, yaw in zip(indice, positions, yaws):
        padded_position_tensor = -torch.ones((79, position.shape[-1]))
        padded_position_tensor[idx] = position
        padded_positions.append(padded_position_tensor)

        padded_yaw_tensor = -torch.ones((79,))
        padded_yaw_tensor[idx] = yaw
        padded_yaws.append(padded_yaw_tensor)

    positions_tensor = torch.stack(padded_positions, dim=1)
    yaws_tensor = torch.stack(padded_yaws, dim=1)

    print(f'Positions: {positions_tensor.shape}')
    print(f'Yaws: {yaws_tensor.shape}')

    # frames x objects x position, yaw, ...
    obj_location_data = torch.zeros((positions_tensor.shape[0], n_objects, 6))

    obj_location_data[..., :3] = positions_tensor
    obj_location_data[..., 3] = yaws_tensor

    # TODO: Is this correct? How does the position in the tracklets/ position translate to the obj metadata/ the object?
    obj_location_data[..., 4] = torch.zeros((positions_tensor.shape[0], n_objects))

    for i in range(n_objects):
        obj_location_data[indice[i], i, 4] = i + 1

    obj_location_data = obj_location_data.reshape(
        obj_location_data.shape[0], 1, 1, n_objects*2, 3)

    return obj_location_data, n_objects


def get_object_models_from_other_scenes(config_path):
    """Get object models from other scenes."""

    scene_config_manager = scm.SceneConfigManager(
        'synthetic_data_generation/scene_configs_decent_miraculix.yaml')

    object_model_ids_from_other_scenes = oms.get_object_model_ids_from_other_scenes(
        scene_config_manager, config_path)

    return object_model_ids_from_other_scenes



def get_best_object_models_for_tracklets(pipeline, angular_embeddings_tracklets, models_from_other_scenes: list[oms.ObjectModelsIDsFromOtherScene], select_object_model_weights: list[int]):
    '''Get the best object models for given tracklets.'''

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
    
    object_dimensions_list = []
    object_model_keys_with_scene = []

    print(f'Diff matrix: {diff_matrix.shape}: {diff_matrix}')

    for diffs_per_trajectory in diff_matrix:
        if len(diffs_per_trajectory) == 0:
            continue

        sorted_indice = diffs_per_trajectory.argsort()

        # Sample models via index here
        # Bias sampling with decreasing probability for higher differences/ higher indices of indices
        # Get object model id and scene based on selected indice
        # Adjust to sample with decreasing probability for higher indices

        n_best_indice = sorted_indice[:len(select_object_model_weights)]
        n_indice = len(n_best_indice)
        possible_indices = [i for i in range(n_indice)]
        weights = select_object_model_weights[:n_indice]
        print(f'Debug index selection: {n_best_indice}, {n_indice}, {possible_indices}, {weights}')
        selected_index_position = random.choices(possible_indices, weights=weights, k=1)[0]
        selected_index = sorted_indice[selected_index_position].item()

        selected_scene_model_object, selected_model_index_in_scene = index_to_scene_name[selected_index]

        scene_name = selected_scene_model_object.scene_name
        scene_checkpoint = selected_scene_model_object.scene_checkpoint
        object_model_key = selected_scene_model_object.object_model_ids[selected_model_index_in_scene]
        object_dimensions = selected_scene_model_object.object_metadata[selected_model_index_in_scene][1:4]

        object_dimensions_list.append(object_dimensions)
        # print(f'object dimensions: {object_dimensions}')

        object_model_key_with_scene = oms.add_object_model_to_scene_graph(scene_graph=scene_graph, import_scene_name=scene_name, import_scene_checkpoint=scene_checkpoint, object_model_key=object_model_key)
        object_model_keys_with_scene.append(object_model_key_with_scene)

    if len(object_model_keys_with_scene) == 0:
        return None

    obj_key_tensor = torch.tensor(object_model_keys_with_scene)
    obj_dimensions_tensor = torch.stack(object_dimensions_list)

    # Stack tensor and return
    print(f'obj_key_tensor: {obj_key_tensor.shape}')
    print(f'obj_dimensions_tensor: {obj_dimensions_tensor.shape}')

    obj_metadata_tensor = torch.cat((obj_key_tensor.unsqueeze(-1), obj_dimensions_tensor), dim=1)

    return obj_metadata_tensor