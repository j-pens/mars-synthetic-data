from dataclasses import dataclass

@dataclass(init=False)
class BoundingBox:
    """Bounding box class."""
    
    def __init__(self, x, y, z, yaw, dx, dy, dz, class_id, obj_id):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.class_id = class_id
        self.obj_id = obj_id

def generate_kitti_annotation_line(bounding_box):
    annotation_line = f"{bounding_box.x} {bounding_box.y} {bounding_box.z} {bounding_box.dx} {bounding_box.dy} {bounding_box.dz} {bounding_box.yaw} {bounding_box.class_id} {bounding_box.obj_id}"
    return annotation_line