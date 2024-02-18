from nerfstudio.cameras.cameras import Cameras
from object_trajectory_generation import BoundingBoxTracklet, get_bounding_boxes_with_object_ids, get_min_camera_distance, remove_points_object_not_visible

import torch

def manipulate_scene_trajectories(cameras: Cameras, obj_metadata, obj_location_data):
    """Manipulate scene trajectories."""
    
    # get camera trajectory
    cam2world = cameras.camera_to_worlds
    times = cameras.times
    
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


    bounding_box_tracklets = get_bounding_boxes_with_object_ids(batch_objects_dyn=obj_location_data, obj_metadata=obj_metadata)

    bounding_box_tracklet_keys = list(bounding_box_tracklets.keys())

    bounding_box_tracklets_list = list(bounding_box_tracklets.values())

    for tracklet in bounding_box_tracklets_list:
        # tracklet.save(f'pandaset_tracklets/seq_011_corrected_axes/bounding_box_tracklet_{tracklet.obj_id}.pt')

        remove_points_object_not_visible(tracklet=tracklet)

        # tracklet.x += 0.5e-2 # 0.5m

        # random_index = torch.randint(0, len(bounding_box_tracklets_list), (1,)).item()

        # tracklet.obj_model_id = bounding_box_tracklet_keys[random_index]

    index = 0

    keys_to_dists = {key: get_min_camera_distance(bounding_box_tracklets[key], cam2worlds=cam2world) for key in bounding_box_tracklet_keys}

    print(keys_to_dists)

    sorted_keys_asc_dist = sorted(bounding_box_tracklet_keys, key=lambda x: keys_to_dists[x])

    print(sorted_keys_asc_dist)

    print(sorted_keys_asc_dist[index])

    n_closest_objects = 2

    random_indices = torch.randint(0, n_closest_objects, (len(bounding_box_tracklet_keys),))
    random_obj_model_ids = torch.tensor([sorted_keys_asc_dist[i] for i in random_indices])

    obj_metadata[1:, 0] = random_obj_model_ids

    # exit()



