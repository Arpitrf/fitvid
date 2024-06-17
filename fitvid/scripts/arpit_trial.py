from fitvid.utils.utils import edit_and_merge_hdf5, partition_dataset_train_valid
import h5py
import matplotlib.pyplot as plt
import numpy as np

# edit_and_merge_hdf5('/home/arpit/test_projects/OmniGibson/backup/dataset.hdf5', '/home/arpit/test_projects/OmniGibson/backup/dataset_new.hdf5')

# partition_dataset_train_valid('/home/arpit/test_projects/OmniGibson/backup/dataset2.hdf5')

f = h5py.File('/home/arpit/test_projects/OmniGibson/dynamics_model_dataset/dataset.hdf5', "r") 
# print("actions: ", np.array(f['data/episode_00021/actions/actions']))
print(f['data/episode_00000']['extras'].keys())
# demos = f["data"].keys()
# print("demos: ", demos)
# print("f[masks]: ", f['mask']['train'])
# print("f[masks]: ", f['mask']['valid'])
# print(type(np.array(f['mask']['valid'])[0]))
# print("attrs: ", f['data']['episode_00466'].attrs.keys())
# for k in f['data']['episode_00466']['proprioceptions'].keys():
#     print(f['data']['episode_00466']['proprioceptions'][k])

# plt.imshow(f['data']['episode_00466']['observations']['rgb'][0,:,:,:3])
# plt.show()


succ_episodes = 0
last_step_grasped = 0
for k in f['data'].keys():
    grasps = np.array(f['data'][k]['extras']['grasps'])
    # print("len(grasps): ", grasps.shape)
    if any(grasps):
        succ_episodes += 1
        if not grasps[-1]:
            print("k, grasps: ", k, grasps)

print("succ episodes: ", succ_episodes)
print("episodes with last step as grasped: ", last_step_grasped)

rgb = f['data/episode_00025/observations/rgb']
print("rgb.shape:" , rgb.shape)

# fig, ax = plt.subplots(3, 6)
# ax[0][0].imshow(rgb[0, :, :, :3])
# ax[0][1].imshow(rgb[1, :, :, :3])
# ax[0][2].imshow(rgb[2, :, :, :3])
# ax[0][3].imshow(rgb[3, :, :, :3])
# ax[0][4].imshow(rgb[4, :, :, :3])
# ax[0][5].imshow(rgb[5, :, :, :3])
# ax[1][0].imshow(rgb[6, :, :, :3])
# ax[1][1].imshow(rgb[7, :, :, :3])
# ax[1][2].imshow(rgb[8, :, :, :3])
# ax[1][3].imshow(rgb[9, :, :, :3])
# ax[1][4].imshow(rgb[10, :, :, :3])
# ax[1][5].imshow(rgb[11, :, :, :3])
# ax[2][0].imshow(rgb[12, :, :, :3])
# ax[2][1].imshow(rgb[13, :, :, :3])
# ax[2][2].imshow(rgb[14, :, :, :3])
# ax[2][3].imshow(rgb[15, :, :, :3])
# ax[2][4].imshow(rgb[16, :, :, :3])
# ax[2][5].imshow(rgb[17, :, :, :3])
# plt.show()