import pandaset as ps
from pandaset.sequence import Sequence
from transformers import pipeline
from PIL import Image
from tqdm import tqdm   # Progress bar
import os


checkpoint = "vinvino02/glpn-nyu"
depth_estimator = pipeline("depth-estimation", model=checkpoint, device=0)

# path = "/zfs/penshorn/master_thesis/datasets/raw/PandaSet/011/camera/front_camera/00.jpg"
# image = Image.open(path)

# predictions = depth_estimator(image)
# print(type(predictions["depth"]))


pandaset_root = "/zfs/penshorn/master_thesis/datasets/raw/PandaSet"
dataset = ps.DataSet(pandaset_root)

sequence_names = dataset.sequences()
print(sequence_names, len(sequence_names))

for sequence_name in tqdm(sequence_names, desc="Processing sequences"):
    sequence: Sequence = dataset[sequence_name].load_camera()

    sequence_dir = os.path.join(pandaset_root, sequence_name)
    cameras = sequence.camera

    for camera_name in tqdm(cameras.keys(), desc="Processing cameras"):
        camera = cameras[camera_name]
        camera_path = os.path.join(sequence_dir, 'camera', camera_name)
        depth_camera_dir = os.path.join(sequence_dir, 'monocular_depth', camera_name)
        os.makedirs(depth_camera_dir, exist_ok=True)

        for frame_index, frame in tqdm(enumerate(camera), desc='Processing frames', total=80):
            depth_filename = str(frame_index).zfill(2) + '.png'
            prediction = depth_estimator(frame)
            prediction['depth'].save(os.path.join(depth_camera_dir, depth_filename))
            
    dataset.unload(sequence=sequence_name)