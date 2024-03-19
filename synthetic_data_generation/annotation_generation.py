from calendar import c
import stat
import pandaset as ps
from object_trajectory_generation import BoundingBoxTracklet
from uuid import uuid4
import numpy as np
import pandas as pd


# I didn't put pedestrian, many other classes are available: https://github.com/scaleapi/pandaset-devkit/blob/master/docs/annotation_instructions_cuboids.pdf
_sem2label_pandaset = {"Car": 0, "Pickup Truck": 0, "Medium-sized Truck": 2, "Semi-truck": 2, "Tram / Subway": 3, "Train": 3, "Trolley": 3, "Bus": 3}

label2sem_pandaset = {0: "Car", 2: "Medium-sized Truck", 3: "Tram / Subway"}


class SyntheticPandaSetAnnotationGenerator:

    def __init__(self, dataset_root_path: str):
        self.dataset_root_path = dataset_root_path
        self.dataset = ps.DataSet(self.dataset_root_path)
        self.sequences = self.dataset.sequences()
        self._uuids = set()


    def get_static_cuboids(self, original_sequence_name: str, frames: tuple[int, int] = (0, 80)):
        """Get cuboids/ bounding boxes of static objects/ scene elements."""

        sequence: ps.Sequence = self.dataset[original_sequence_name]

        sequence.load_cuboids()

        cuboids_list_of_dfs = sequence.cuboids.data

        list_of_filtered_dfs = []
        for df in cuboids_list_of_dfs:
            filtered_df = df[df['stationary'] == True]
            list_of_filtered_dfs.append(filtered_df)
            self._uuids.update(filtered_df['uuid'].tolist())

        assert len(list_of_filtered_dfs) == 80

        list_of_filtered_dfs = list_of_filtered_dfs[frames[0]:frames[1]]

        return list_of_filtered_dfs, cuboids_list_of_dfs
    
    def create_dynamic_cuboids(self, bounding_box_tracklets: list[BoundingBoxTracklet]):
        """Create dynamic cuboids/ bounding boxes of dynamic objects/ scene elements."""
        
        tracklets_by_frames = {}

        for bounding_box_tracklet in bounding_box_tracklets:

            original_indices = bounding_box_tracklet.original_indices
            # TODO: Add class id 2 label name mapping
            label = label2sem_pandaset[bounding_box_tracklet.class_id]
            x = bounding_box_tracklet.x * 100
            y = bounding_box_tracklet.y * 100
            z = bounding_box_tracklet.z * 100
            yaw = bounding_box_tracklet.yaw
            dx = bounding_box_tracklet.dx.item() * 100
            dy = bounding_box_tracklet.dy.item() * 100
            dz = bounding_box_tracklet.dz.item() * 100

            # TODO: Check for camera used for bounding box tracklet
            camera_used = -1

            stationary = False
            object_motion = 'Moving'

            uuid = self.get_uuid()

            for i, original_index in enumerate(original_indices):
                original_index_int = original_index.item()
                if original_index_int not in tracklets_by_frames:
                    tracklets_by_frames[original_index_int] = []

                tracklets_by_frames[original_index_int].append([uuid, label, yaw[i].item(), stationary, camera_used, x[i].item(), y[i].item(), z[i].item() + dz/2, dx, dy, dz, object_motion])

        print(tracklets_by_frames)
        dynamic_cuboids_dict_of_dfs = {}
        for frame_idx, bboxes in tracklets_by_frames.items():
            bboxes_np = np.array(bboxes)
            print(bboxes_np.shape)
            dynamic_cuboids_df = pd.DataFrame(bboxes_np, columns=['uuid', 'label', 'yaw', 'stationary', 'camera_used', 'position.x', 'position.y', 'position.z', 'dimensions.x', 'dimensions.y', 'dimensions.z', 'attributes.object_motion'])
            dynamic_cuboids_dict_of_dfs[frame_idx] = dynamic_cuboids_df

        return dynamic_cuboids_dict_of_dfs


    def merge_static_and_dynamic_cuboids(self, static_cuboids: list[pd.DataFrame], dynamic_cuboids: dict[int, pd.DataFrame]):

        merged_cuboids = []
        for i in dynamic_cuboids.keys():
            static_cuboid = static_cuboids[i]
            dynamic_cuboid = dynamic_cuboids[i]
            merged_cuboid = pd.concat([static_cuboid, dynamic_cuboid], axis=0)
            merged_cuboids.append(merged_cuboid)

        return merged_cuboids


    def get_uuid(self):
        """Get unique identifier for dynamic cuboid."""

        new_uuid = None
        while new_uuid is None or new_uuid in self._uuids:
            new_uuid = uuid4()
        self._uuids.add(new_uuid)

        return new_uuid
            

        


def main():
    dataset_root_path = '/zfs/penshorn/master_thesis/datasets/raw/PandaSet'
    synthetic_pandaset_annotation_generator = SyntheticPandaSetAnnotationGenerator(dataset_root_path=dataset_root_path)

    sequence_name = '001'
    list_of_static_cuboid_dfs = synthetic_pandaset_annotation_generator.get_static_cuboids(original_sequence_name=sequence_name)

    print(list_of_static_cuboid_dfs[0].head())




if __name__ == '__main__':
    main()