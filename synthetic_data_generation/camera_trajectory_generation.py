from nerfstudio.cameras.cameras import Cameras
import torch
import synthetic_data_pipeline_config_manager as sdpcm

# TODO: Create camera trajectory using splines
def get_camera_trajectory(cameras: Cameras, jitter=0.0005):
   """Get camera trajectory."""
   
   cam2world = cameras.camera_to_worlds

   translations = cam2world[:, :, 3]

   translations += torch.randn_like(translations) * jitter

   cam2world[:, :, 3] = translations

   return cameras
