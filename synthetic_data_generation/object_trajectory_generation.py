import torch
from dataclasses import dataclass
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

    print(f'Valid points for tracklet {tracklet.obj_model_id}: {valid_indices}')


    invalid_x = tracklet.x[~remove_invalid_points_condition]
    invalid_y = tracklet.y[~remove_invalid_points_condition]
    invalid_z = tracklet.z[~remove_invalid_points_condition]

    invalid_points = torch.stack((invalid_x, invalid_y, invalid_z), dim=1)

    print(f'Invalid points for tracklet {tracklet.obj_model_id}: {invalid_points.shape}')

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

