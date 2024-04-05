import torch_cubic_spline_grids

trajectory_sampling_jitter = 0.25
spline_optimization_steps = 5000
spline_add_noise_to_observations = False
spline_max_noise = 0.2
spline_grid_class = torch_cubic_spline_grids.CubicCatmullRomGrid1d
print_spline_loss = False
return_spline_optimizer = False
spline_max_control_points = 10
max_acceleration_check = 5 # m/s^2
max_velocity_check = 50 # m/s
n_closest_objects = 5
max_object_distance = 25 # m

camera_jitter = 0.0005 # m
n_bins_histograms = 4096


n_best_object_models = 3
select_object_model_weights = [80, 15, 5]


n_scenes = 2
light_conditions = ['day'] # day, night, frontal_lighting
scene_configs_path = 'synthetic_data_generation/scene_configs_decent_miraculix.yaml'

synthetic_dataset_root = '/zfs/penshorn/master_thesis/datasets/synthetic/PandaSet_0001'
original_dataset_root = '/zfs/penshorn/master_thesis/datasets/raw/PandaSet'