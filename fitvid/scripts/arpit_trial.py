from fitvid.utils.utils import edit_and_merge_hdf5, partition_dataset_train_valid
import h5py
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True, precision=3)
import torch

# edit_and_merge_hdf5('/home/arpit/test_projects/OmniGibson/backup/dataset.hdf5', '/home/arpit/test_projects/OmniGibson/backup/dataset_new.hdf5')

# partition_dataset_train_valid('/home/arpit/test_projects/OmniGibson/backup/dataset.hdf5')

# f = h5py.File('/home/arpit/test_projects/OmniGibson/backup/dataset_new.hdf5', "r") 
f = h5py.File('/home/arpit/test_projects/OmniGibson/dynamics_model_dataset/dataset.hdf5', "r") 
# print(len(f['data'].keys()))
# print(f['mask']['train'])
# print("---", 'data' in f.keys())
# print(f['data/episode_00001/observations'].keys())
# demos = f["data"].keys()
# demo_lens = []
# for i, demo in enumerate(demos):
#     if i != 11:
#         continue
#     print(i, np.array(f[f'data/{demo}/actions/actions']))
#     input()

succ_episodes = 0
for k in f['data'].keys():
    grasps = np.array(f['data'][k]['extras']['grasps'])
    # print("len(grasps): ", grasps.shape)
    # print("k, grasps: ", k, grasps)
    # print("----------------")
    if any(grasps):
        succ_episodes += 1
print("succ episodes: ", succ_episodes)

# sorted_demo_lens = sorted(demo_lens)
# print("sorted_demo_len: ", sorted_demo_lens)
# print("demos: ", len(demos))
# print("f[masks]: ", f['mask']['train'])
# print("f[masks]: ", f['mask']['valid'])
# print(type(np.array(f['mask']['valid'])[0]))
# print("attrs: ", f['data']['episode_00466'].attrs.keys())
# for k in f['data']['episode_00466']['proprioceptions'].keys():
#     print(f['data']['episode_00466']['proprioceptions'][k])

# plt.imshow(f['data']['episode_00466']['observations']['rgb'][0,:,:,:3])
# plt.show()


# pred = torch.tensor([
#         [0.0, 0.0, 0.0, 0.9, 1.0, 1.0, 1.0],
#         [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
# ])
# target = torch.tensor([
#         [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
#         [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
# ])
# # pred = torch.tensor(0.9)
# # target = torch.tensor(0.0)
# pred = pred.to(torch.float64)
# target = target.to(torch.float64)

# import torch.nn as nn
# bce_error = nn.BCELoss()(pred, target)
# print("bce_error: ", bce_error)