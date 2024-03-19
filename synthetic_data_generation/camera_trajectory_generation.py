from nerfstudio.cameras.cameras import Cameras
import torch

def get_camera_trajectory(cameras: Cameras):
   """Get camera trajectory."""
   
   cam2world = cameras.camera_to_worlds

   translations = cam2world[:, :, 3]

   # 0.5 m jitter
   jitter = 0.0005
   translations += torch.randn_like(translations) * jitter

   cam2world[:, :, 3] = translations

   return cameras
