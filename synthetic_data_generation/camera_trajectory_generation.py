from nerfstudio.cameras.cameras import Cameras

def get_camera_trajectory(cameras: Cameras):
   """Get camera trajectory."""
   
   cam2world = cameras.camera_to_worlds

   times = cameras.times