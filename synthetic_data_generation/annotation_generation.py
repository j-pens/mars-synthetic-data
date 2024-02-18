from object_trajectory_generation import BoundingBoxTracklet


def generate_kitti_annotation_line(bounding_box_tracklet: BoundingBoxTracklet) -> list[str]:
    annotation_line = f"{bounding_box_tracklet.x} {bounding_box_tracklet.y} {bounding_box_tracklet.z} {bounding_box_tracklet.dx} {bounding_box_tracklet.dy} {bounding_box_tracklet.dz} {bounding_box_tracklet.yaw} {bounding_box_tracklet.class_id} {bounding_box_tracklet.obj_id} {bounding_box_tracklet.obj_model_id}"
    return annotation_line