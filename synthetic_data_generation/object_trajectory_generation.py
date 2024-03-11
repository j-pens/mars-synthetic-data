import torch
from dataclasses import dataclass
from torch_cubic_spline_grids import CubicCatmullRomGrid1d
from torch_cubic_spline_grids._base_cubic_grid import CubicSplineGrid
from typing import Union, Tuple
@dataclass(init=False)
class BoundingBoxTracklet():
    """Bounding box class."""
    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor
    yaw: torch.Tensor
    dx: torch.Tensor
    dy: torch.Tensor
    dz: torch.Tensor
    class_id: int
    obj_id: int
    obj_model_id: int
    original_indices: torch.Tensor
    tracklet_to_meters_factor: int

    def __init__(self, x, y, z, yaw, dx, dy, dz, class_id, obj_id, obj_model_id):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.class_id = class_id
        self.obj_id = obj_id
        self.obj_model_id = obj_model_id
        self.original_indices = torch.arange(x.shape[0])
        self.tracklet_to_meters_factor = 100

        # No real dataclass usage for now, let's see if we need it at some point
        # self.__post_init__()  # This will do the dataclass post_init and broadcast all the tensors

    def save(self, path):
        """Save bounding box."""
        torch.save(self, path)


    def __str__(self):
        return f"Bounding box: object_model_id: {self.obj_model_id}, x: {self.x}, y: {self.y}, z: {self.z}, yaw: {self.yaw} dx: {self.dx}, dy: {self.dy}, dz: {self.dz}, class_id: {self.class_id}, obj_id: {self.obj_id}"


def get_bounding_boxes_with_object_ids(batch_objects_dyn, obj_metadata) -> dict[int, BoundingBoxTracklet]:
    """Get tracklets with object ids."""
    
    # first row is constant, and irrelevant, so I remove it
    obj_metadata = obj_metadata[1:, :]

    # object indice in the full metadata, including the first row
    obj_idx = batch_objects_dyn[..., 4]

    # ids of the object models, i.e. the object NeRFs
    object_model_ids = obj_metadata[:, 0]

    # dimensions of the object bounding boxes, i.e. dx, dy, dz
    dimensions = obj_metadata[:, 1:4]

    # class ids of the objects, important for class-wise object representations
    class_ids = obj_metadata[:, 4]

    tracklets = {}
    for i, obj_id in enumerate(object_model_ids):
        obj_model_id = int(obj_id.item())
        batch_objects_dyn_row = batch_objects_dyn[..., i, :]

        # position of the object: x, y, z
        pos = batch_objects_dyn_row[..., :3]
        
        xs = pos[..., 0]
        ys = pos[..., 1]
        zs = pos[..., 2]

        yaw = batch_objects_dyn_row[..., 3]

        dimensions_obj = dimensions[i]

        dxs = dimensions_obj[0]
        dys = dimensions_obj[1]
        dzs = dimensions_obj[2]

        # obj_index references based on the index in the complete object metadata, including the first row
        obj_index = int(obj_idx[0, i].item())

        class_id = int(class_ids[i].item())

        tracklets[obj_model_id] = BoundingBoxTracklet(xs, ys, zs, yaw, dxs, dys, dzs, class_id, obj_index, obj_model_id)

    return tracklets


def remove_points_object_not_visible(tracklet: BoundingBoxTracklet):

    # points are -1, -1, -1 if the object is not visible in the frame

    remove_invalid_points_condition = (tracklet.x != tracklet.y) | (tracklet.x * tracklet.tracklet_to_meters_factor != -1) | (tracklet.y * tracklet.tracklet_to_meters_factor != -1)

    # Save valid indices for velocity and acceleration calculations
    valid_indices = tracklet.original_indices[remove_invalid_points_condition]
    tracklet.original_indices = valid_indices

    # print(f'Valid points for tracklet {tracklet.obj_model_id}: {valid_indices}')


    invalid_x = tracklet.x[~remove_invalid_points_condition]
    invalid_y = tracklet.y[~remove_invalid_points_condition]
    invalid_z = tracklet.z[~remove_invalid_points_condition]

    invalid_points = torch.stack((invalid_x, invalid_y, invalid_z), dim=1)

    # print(f'Invalid points for tracklet {tracklet.obj_model_id}: {invalid_points.shape}')

    # TODO: Extend for other tensors in tracklet
    tracklet.x = tracklet.x[remove_invalid_points_condition]
    tracklet.y = tracklet.y[remove_invalid_points_condition]
    tracklet.z = tracklet.z[remove_invalid_points_condition]
    tracklet.yaw = tracklet.yaw[remove_invalid_points_condition]


def get_min_camera_distance(tracklet: BoundingBoxTracklet, cam2worlds: torch.Tensor):
    """Get minimum camera distance."""

    points = torch.stack((tracklet.x * tracklet.tracklet_to_meters_factor, tracklet.y * tracklet.tracklet_to_meters_factor, tracklet.z * tracklet.tracklet_to_meters_factor), dim=1)

    translations = cam2worlds[:, :, 3]

    camera_xs = translations[:, 0] * tracklet.tracklet_to_meters_factor
    camera_ys = translations[:, 1] * tracklet.tracklet_to_meters_factor
    camera_zs = translations[:, 2] * tracklet.tracklet_to_meters_factor

    camera_positions = torch.stack((camera_xs, camera_ys, camera_zs), dim=1)

    distances = torch.norm(points - camera_positions[tracklet.original_indices], dim=1)

    min_distance = torch.min(distances)

    return min_distance



def sample_with_jitter(n, lower=0, upper=1.0, jitter=0.25):
    '''Sample n points from n equidistant bins between lower and upper with jitter in percent of the bin width.'''
    bin_width = (upper - lower) / n
    bin_centers = torch.linspace(lower, upper, n)

    # Add jitter to bin centers
    jitters = (torch.rand(n) - 0.5) * bin_width * jitter
    samples = bin_centers + jitters

    # Ensure samples are within bounds
    samples = torch.clamp(samples, lower, upper)

    return samples



def get_parametrization(tracklet, optimization_steps=5000, add_noise=False, noise_level=0.2, spline_grid_class=CubicCatmullRomGrid1d, print_loss=False, with_optimizer=False, resolution=10) -> Union[Tuple[CubicSplineGrid, torch.optim.Optimizer], CubicSplineGrid]:
    
    # Limit to N_CONTROL_POINTS, or half of the tracklet length, whichever is smaller, but at least 2
    resolution = max(min(resolution, tracklet.x.shape[0]//7), 2)

    grid_3d = spline_grid_class(resolution=resolution, n_channels=3)
    optimiser = torch.optim.Adam(grid_3d.parameters(), lr=0.05)

    for i in range(optimization_steps):

        x, y = make_observations_on_tracklet(tracklet=tracklet, n=100, add_noise=add_noise, noise_level=noise_level)

        # print(y.shape)

        y = y.squeeze(-1)

        # print(x.shape)

        prediction = grid_3d(x).squeeze()

        # print(prediction.shape)

        optimiser.zero_grad()
        loss = torch.sum((prediction - y)**2)**0.5

        # print(loss)

        loss.backward()
        optimiser.step()
        if print_loss and i % 100 == 0:
            print(loss.item())

    if with_optimizer:
        return grid_3d, optimiser
    else:
        return grid_3d


def calculate_yaw(positions: torch.Tensor):
    '''Calculate yaw from positions'''

    # Calculate facing vectors
    facing_vectors = torch.diff(positions[:, :2], dim=0)

    # Repeat last facing vector for last point
    facing_vectors = torch.cat((facing_vectors, facing_vectors[-1, :].unsqueeze(0)), dim=0)

    # Normalize facing vectors
    facing_vectors = facing_vectors / torch.norm(facing_vectors, dim=1).unsqueeze(-1)

    # print(facing_vectors)

    # Calculate yaw
    yaws = torch.atan2(facing_vectors[:, 1], facing_vectors[:, 0])

    return yaws



def remove_physically_implausible_points(tracklet: BoundingBoxTracklet):

    MAX_ACCELERATION = 5 # m/s^2, being very generous here, F1 car on average 14.2 m/s^2
    MAX_VELOCITY = 50 # m/s, roughly 180

    velocities, accelerations = get_dynamics(tracklet=tracklet)

    # print(velocities.shape, accelerations.shape)

    implausible_velocities_mask = velocities > MAX_VELOCITY
    implausible_accelerations_mask = accelerations > MAX_ACCELERATION

    # print(implausible_velocities_mask.shape)
    # print(implausible_accelerations_mask.shape)

    # implausible_velocities_mask = torch.cat((torch.zeros((1), dtype=torch.bool), implausible_velocities_mask))
    # implausible_accelerations_mask = torch.cat((torch.zeros((2), dtype=torch.bool), implausible_accelerations_mask))

    # print(implausible_velocities_mask.shape)
    # print(implausible_accelerations_mask.shape)

    # print(implausible_velocities_mask.dtype)

    implausible_points_mask = implausible_velocities_mask | implausible_accelerations_mask

    implausible_indices = torch.nonzero(implausible_points_mask)

    # print(implausible_indices)

    tracklet.x = tracklet.x[~implausible_points_mask]
    tracklet.y = tracklet.y[~implausible_points_mask]
    tracklet.z = tracklet.z[~implausible_points_mask]
    tracklet.yaw = tracklet.yaw[~implausible_points_mask]
    tracklet.original_indices = tracklet.original_indices[~implausible_points_mask]


    # TODO: Extend to remove physically implausible points, e.g. large velocity or acceleration

# TODO: Extend to use higher order diffs to remove less points 
# -> 1st order diff high for neighbor of outlier as well
def get_dynamics(tracklet: BoundingBoxTracklet) -> tuple[torch.Tensor, torch.Tensor]:

    points = torch.stack((tracklet.x, tracklet.y, tracklet.z), dim=1)

    # print(points.shape)

    velocity_vectors_raw = torch.diff(points, dim=0, append=points[-2, :].unsqueeze(0))

    index_differences = torch.diff(tracklet.original_indices, append=tracklet.original_indices[-2].unsqueeze(0)).unsqueeze(-1)

    # print(f'Index differences for tracklet {tracklet.obj_model_id}: {index_differences}')

    delta_t = 0.1 # 10 Hz sampling frequency in PandaSet dataset

    velocity_vectors_timed = velocity_vectors_raw / (torch.abs(index_differences) * delta_t)

    # velocity_vectors_length = torch.norm(velocity_vectors_raw, dim=1)
    velocity_vectors_length_timed = torch.norm(velocity_vectors_timed, dim=1)

    # print('Velocity vectors')
    # print(velocity_vectors_raw.shape)

    # print('Velocity (length)')
    # print(velocity_vectors_length.shape)
    # print(velocity_vectors_length)

    # print('Velocity timed length')
    # print(velocity_vectors_length_timed.shape)
    # print(velocity_vectors_length_timed)

    acceleration_vectors = torch.diff(velocity_vectors_timed, dim=0, append=velocity_vectors_timed[-2, :].unsqueeze(0))
    acceleration_vectors_length = torch.norm(acceleration_vectors, dim=1)

    # print('Acceleration vectors')
    # print(acceleration_vectors_length.shape)
    # print(acceleration_vectors_length)

    return velocity_vectors_length_timed, acceleration_vectors_length


def make_observations_on_tracklet(tracklet, n, add_noise=False, noise_level=0.2):

    sample_idx = torch.randint(low=0, high=len(tracklet.x), size=(n, 1))

    x = tracklet.x[sample_idx]
    y = tracklet.y[sample_idx]
    z = tracklet.z[sample_idx]

    if add_noise:
        x += noise_level * torch.randn_like(x)
        z += noise_level * torch.randn_like(y)
    
    # should be the right order, l, w, h -> x,z,y
    point = torch.stack((x, y, z), dim=1)

    # print(point.shape)

    return sample_idx/(len(tracklet.x)-1), point 


def get_closest_object_model_ids(bounding_box_tracklets, cam2worlds, n_closest_objects=5):
    """Get the n_closest_objects object models based on the minimum distance to the camera at any point in the sequence."""

    # TODO: Adjust to handle sequences with fewer than n_closest_objects objects correctly

    bounding_box_tracklet_keys = list(bounding_box_tracklets.keys())

    keys_to_dists = {key: get_min_camera_distance(bounding_box_tracklets[key], cam2worlds=cam2worlds) for key in bounding_box_tracklet_keys}

    sorted_keys_asc_dist = sorted(bounding_box_tracklet_keys, key=lambda x: keys_to_dists[x])

    return sorted_keys_asc_dist[:n_closest_objects]