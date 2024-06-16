from fitvid.utils.utils import edit_and_merge_hdf5, partition_dataset_train_valid
import h5py
import matplotlib.pyplot as plt
import numpy as np

# edit_and_merge_hdf5('/home/arpit/test_projects/OmniGibson/backup/dataset.hdf5', '/home/arpit/test_projects/OmniGibson/backup/dataset2.hdf5')

# partition_dataset_train_valid('/home/arpit/test_projects/OmniGibson/backup/dataset2.hdf5')

f = h5py.File('/home/arpit/test_projects/OmniGibson/backup/dataset2.hdf5', "r") 
print(f.keys())
demos = f["data"].keys()
# print("demos: ", demos)
print("f[masks]: ", f['mask']['train'])
print("f[masks]: ", f['mask']['valid'])
print(type(np.array(f['mask']['valid'])[0]))
# print("attrs: ", f['data']['episode_00466'].attrs.keys())
# for k in f['data']['episode_00466']['proprioceptions'].keys():
#     print(f['data']['episode_00466']['proprioceptions'][k])

# plt.imshow(f['data']['episode_00466']['observations']['rgb'][0,:,:,:3])
# plt.show()