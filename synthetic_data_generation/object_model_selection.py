from typing import Literal, Tuple, Union, List

from scene_config_manager import SceneConfigManager, SceneConfig
from model_library import MarsPipelineCheckpoint
import object_trajectory_generation as otg
import torch
import os
from collections import namedtuple
import random
import numpy as np
import synthetic_data_pipeline_config_manager as sdpcm

ObjectModelsIDsFromOtherScene = namedtuple('ObjectModelsIDsFromOtherScene', ['scene_name', 'scene_checkpoint', 'object_model_ids', 'bounding_box_tracklets', 'object_metadata', 'cam2worlds'])

def add_object_model_to_scene_graph(scene_graph, import_scene_name: str, import_scene_checkpoint: MarsPipelineCheckpoint, object_model_key: str):
    """Add object model to scene."""

    object_model_state = import_scene_checkpoint.get_object_model_state(object_model_key)

    # print(f'object_model_state: {object_model_state.keys()}')

    if len(object_model_state) == 0:
        print(f'Object model {object_model_key} not found in checkpoint')
        return scene_graph
    
    # Update object_model_key with original_scene_name to avoid name conflicts
    object_model_key_number = int(object_model_key)

    original_scene_number = int(import_scene_name)

    new_object_id = int(f'{original_scene_number}{str(object_model_key_number).zfill(4)}')

    object_model_key_with_scene = f"object_{original_scene_number}{str(object_model_key_number).zfill(4)}"

    object_model_config = import_scene_checkpoint.get_object_model_template_config()

    scene_graph.object_models[object_model_key_with_scene] = object_model_config.setup(
        scene_box=scene_graph.scene_box,
        num_train_data=scene_graph.num_train_data,
        object_meta=scene_graph.object_meta,
        obj_feat_dim=0,
        car_latents=None,
        car_nerf_state_dict_path=None,
    )

    scene_graph.object_models[object_model_key_with_scene].load_state_dict(object_model_state)

    return new_object_id



def get_object_models_from_other_scenes(scene_config_manager: SceneConfigManager, original_scene_config_path: str, synthetic_data_pipeline_config: sdpcm.SyntheticDataPipelineConfig):
    '''Get object models from other scenes.'''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_scene_config = scene_config_manager.get_scene_config(original_scene_config_path)

    light_condition = original_scene_config.light_condition

    if synthetic_data_pipeline_config.swap_object_models_within_scene:
        compatible_scene_configs = scene_config_manager.get_scene_configs_filtered(light_condition=light_condition, method_name='mars-pandaset-nerfacto-object-wise-recon', scene_name=original_scene_config.scene_name)
    else:
        compatible_scene_configs = scene_config_manager.get_scene_configs_filtered(light_condition=light_condition, method_name='mars-pandaset-nerfacto-object-wise-recon', exclude_scenes=[original_scene_config.scene_name])

    random.shuffle(compatible_scene_configs)

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
            -1 , # datamanager.dataparser.config.max_input_objects,
            datamanager.dataparser.config.add_input_rows * 3
        )
        print(f'obj_location_data_dyn shape: {obj_location_data_dyn.shape}')


        # Object metadata consistent over all frames/ cameras
        obj_metadata = train_dataset.metadata["obj_metadata"]
        print(f'obj_metadata shape: {obj_metadata.shape}')

        # Check distances and filter object models similar to scene_manipulation script
        object_model_id_list, bounding_box_tracklets = otg.get_bounding_boxes_with_object_ids(batch_objects_dyn=obj_location_data_dyn, obj_metadata=obj_metadata)

        # for tracklet in bounding_box_tracklets:
        #     otg.remove_points_object_not_visible(tracklet=tracklet)

        bounding_box_tracklets_min_frames = [bounding_box_tracklet for bounding_box_tracklet in bounding_box_tracklets if bounding_box_tracklet.original_indices.shape[0] > synthetic_data_pipeline_config.n_frames_min_objects]

        object_model_id_list = [bounding_box_tracklet.obj_model_id for bounding_box_tracklet in bounding_box_tracklets_min_frames]
        # angular_bins_tracklets = get_angular_bins_tracklets(tracklets=bounding_box_tracklets.values(), cam2worlds=scene_cameras.camera_to_worlds, n_bins=32)

        print(f'Removed {len(bounding_box_tracklets) - len(bounding_box_tracklets_min_frames)} bounding box tracklets with less than {synthetic_data_pipeline_config.n_frames_min_objects} frames')
        print(f'Remaining {len(bounding_box_tracklets_min_frames)} bounding box tracklets')
        # print([tracklet.original_indices.shape for tracklet in bounding_box_tracklets.values()])
        # print(angular_bins_tracklets)
    
        closest_model_ids, filtered_bounding_box_tracklets = otg.get_closest_object_model_ids(tracklets=bounding_box_tracklets_min_frames, ids=object_model_id_list, cam2worlds=scene_cameras.camera_to_worlds, n_closest_objects=synthetic_data_pipeline_config.n_closest_objects)

        print(config.scene_name)

        filtered_object_metadata = [obj_metadata[tracklet.obj_id, :] for tracklet in filtered_bounding_box_tracklets]

        object_models_ids_from_other_scenes.append(ObjectModelsIDsFromOtherScene(scene_name=config.scene_name, object_model_ids=closest_model_ids, scene_checkpoint=checkpoint, bounding_box_tracklets=filtered_bounding_box_tracklets, object_metadata=filtered_object_metadata, cam2worlds=scene_cameras.camera_to_worlds))

        # if len(object_models_ids_from_other_scenes) >= 2:
        #     break

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

    

def select_object_model_ids(scene_graph):

    object_model_ids = list(scene_graph.object_models.keys())

    object_model_ids.sort(key=lambda x: len(x), reverse=True)

    print(object_model_ids)

    object_model_ids_num_only = [int(obj_model_id[len('object_'):]) for obj_model_id in object_model_ids[:min(5, len(object_model_ids))]]

    print(object_model_ids_num_only)

    return object_model_ids_num_only


def get_angular_bins_tracklets(tracklets, cam2worlds, n_bins=16):
    '''Get angular bins for tracklets.'''

    print(f'Type of tracklets: {type(tracklets)}')

    hists = []

    for tracklet in tracklets:
        
        print(f'Type of tracklet: {type(tracklet)}')

        translations = cam2worlds[:, :, 3]
        points = torch.stack((tracklet.x * tracklet.tracklet_to_meters_factor, tracklet.y * tracklet.tracklet_to_meters_factor, tracklet.z * tracklet.tracklet_to_meters_factor), dim=1)

        # print(f'Points shape: {points.shape}')

        assert tracklet.original_indices.shape[0] == tracklet.x.shape[0] <= 80, f'Tracklet ID: {tracklet.obj_model_id}, Original indices: {tracklet.original_indices} with shape {tracklet.original_indices.shape}, x: {tracklet.x} with shape {tracklet.x.shape}'

        camera_xs = translations[:, 0] * tracklet.tracklet_to_meters_factor
        camera_ys = translations[:, 1] * tracklet.tracklet_to_meters_factor
        camera_zs = translations[:, 2] * tracklet.tracklet_to_meters_factor

        camera_positions = torch.stack((camera_xs, camera_ys, camera_zs), dim=1)

        # print(camera_positions.shape)

        object2cam_translation = camera_positions[tracklet.original_indices, :2] - points[:, :2]
        object2cam_translation_norms = torch.linalg.vector_norm(object2cam_translation, dim=1, keepdim=True, ord=2)

        object2cam_directions = object2cam_translation / object2cam_translation_norms

        object_forward_vectors = torch.stack((torch.cos(tracklet.yaw), torch.sin(tracklet.yaw)), dim=1)
        object_forward_norms = torch.linalg.vector_norm(object_forward_vectors, dim=1, keepdim=True, ord=2)

        object_forward_directions = object_forward_vectors / object_forward_norms

        dot_products = torch.sum(object2cam_directions * object_forward_directions, dim=1)

        # print(dot_products.shape)

        angles = torch.acos(dot_products / (object2cam_translation_norms.squeeze(-1) * object_forward_norms.squeeze(-1)))

        # print(angles.shape)

        # print(angles.max().item(), angles.min().item())

        # embed into 2D space to get rid of 0-2pi discontinuity
        angular_embedding = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)

        angular_embedding_np = angular_embedding.numpy()

        # split into bins
        hist_np = np.histogram2d(angular_embedding_np[:, 0], angular_embedding_np[:, 1], bins=n_bins, range=[[-1, 1], [-1, 1]], density=False)[0]

        hist = torch.from_numpy(hist_np)

        hists.append(hist)

    return hists



def get_angular_embeddings_tracklets(tracklets, cam2worlds, n_bins=16):
    '''Get angular bins for tracklets.'''

    print(f'Type of tracklets: {type(tracklets)}')

    angular_embeddings = []

    for tracklet in tracklets:
        
        print(f'Type of tracklet: {type(tracklet)}')

        translations = cam2worlds[:, :, 3]
        points = torch.stack((tracklet.x * tracklet.tracklet_to_meters_factor, tracklet.y * tracklet.tracklet_to_meters_factor, tracklet.z * tracklet.tracklet_to_meters_factor), dim=1)

        # print(f'Points shape: {points.shape}')

        assert tracklet.original_indices.shape[0] == tracklet.x.shape[0] <= 80, f'Tracklet ID: {tracklet.obj_model_id}, Original indices: {tracklet.original_indices} with shape {tracklet.original_indices.shape}, x: {tracklet.x} with shape {tracklet.x.shape}'

        camera_xs = translations[:, 0] * tracklet.tracklet_to_meters_factor
        camera_ys = translations[:, 1] * tracklet.tracklet_to_meters_factor
        camera_zs = translations[:, 2] * tracklet.tracklet_to_meters_factor

        camera_positions = torch.stack((camera_xs, camera_ys, camera_zs), dim=1)

        # print(camera_positions.shape)

        object2cam_translation = camera_positions[tracklet.original_indices, :2] - points[:, :2]
        object2cam_translation_norms = torch.linalg.vector_norm(object2cam_translation, dim=1, keepdim=True, ord=2)

        object2cam_directions = object2cam_translation / object2cam_translation_norms

        object_forward_vectors = torch.stack((torch.cos(tracklet.yaw), torch.sin(tracklet.yaw)), dim=1)
        object_forward_norms = torch.linalg.vector_norm(object_forward_vectors, dim=1, keepdim=True, ord=2)

        object_forward_directions = object_forward_vectors / object_forward_norms

        dot_products = torch.sum(object2cam_directions * object_forward_directions, dim=1)

        # print(dot_products.shape)

        angles = torch.acos(dot_products / (object2cam_translation_norms.squeeze(-1) * object_forward_norms.squeeze(-1)))

        # print(angles.shape)

        # print(angles.max().item(), angles.min().item())

        # embed into 2D space to get rid of 0-2pi discontinuity
        angular_embedding = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)

        angular_embeddings.append(angular_embedding)

    return angular_embeddings

def get_angular_embeddings_means(angular_embeddings):
    '''Get means of angular embeddings.'''

    means = []
    for angular_embedding in angular_embeddings:

        mean = torch.mean(angular_embedding, dim=0)

        assert mean.shape == (2,); f'Mean shape: {mean.shape}'

        means.append(mean)

    return means


def get_mean_diff_matrix(angular_embeddings1, angular_embeddings2):
    '''Get mean diff matrix.'''

    means1 = get_angular_embeddings_means(angular_embeddings=angular_embeddings1)
    means2 = get_angular_embeddings_means(angular_embeddings=angular_embeddings2)

    diff_matrix = torch.zeros(len(means1), len(means2))

    for i in range(len(means1)):
        for j in range(len(means2)):
            diff_matrix[i, j] = torch.sum(torch.abs(means1[i] - means2[j]))

    return diff_matrix

    


def get_angular_bins_stats(histograms):
    '''Get stats from normalized histograms.'''

    mean_indices = []
    for hist in histograms:
        
        # print(f' hist shape: {hist.shape}')

        nonzero_indices = torch.nonzero(hist)


        nonzero_hist_values = hist[nonzero_indices[:, 0], nonzero_indices[:, 1]].unsqueeze(-1)

        # print(f'Nonzero indices hist: {nonzero_hist_values.shape}')
        # print(nonzero_hist_values)
        # print(f'Nonzero indices: {nonzero_indices.shape}')
        # print(nonzero_indices)

        weighted_indices = nonzero_hist_values*nonzero_indices

        # print(f'Weighted indices: {weighted_indices.shape}')
        # print(weighted_indices)

        mean_indice = torch.sum(nonzero_hist_values*nonzero_indices, dim=0) / torch.sum(nonzero_hist_values)

        mean_indices.append(mean_indice)

    return mean_indices


def get_histogram_difference(hists1, hists2):
    '''Get histogram difference (mean).'''

    mean_indices1 = get_angular_bins_stats(histograms=hists1)
    mean_indices2 = get_angular_bins_stats(histograms=hists2)

    diff_matrix = torch.zeros(len(mean_indices1), len(mean_indices2))

    for i in range(len(mean_indices1)):
        for j in range(len(mean_indices2)):
            diff_matrix[i, j] = torch.sum(torch.abs(mean_indices1[i] - mean_indices2[j]))

    # most similar histogram for a given histogram can be found by using the argmin of the row corresponding to the histogram

    return diff_matrix