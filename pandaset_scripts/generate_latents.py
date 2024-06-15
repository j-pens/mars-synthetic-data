import torch

filename = '../pandaset_latents/pandaset_init_seq012_80_frames.pt'
frame_idx_list = list(range(80))
object_id_range = range(0,500)


indices = []
car_latents = []
for frame_idx in frame_idx_list:
    for object_id in object_id_range:
        indices.append({'fid':frame_idx, 'oid': object_id})
        car_latents.append(torch.randn((1, 512), dtype= torch.float32))

car_latents = torch.cat(car_latents)
torch.save({'latents':car_latents, 'indices':indices}, filename)

