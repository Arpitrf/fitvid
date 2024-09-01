import os
import h5py
import numpy as np

from fitvid.utils.utils import pad_sequence

from torchvision.transforms import Resize, InterpolationMode

import torch.utils.data
import matplotlib.pyplot as plt

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hdf5_path,
        obs_keys,
        dataset_keys,
        frame_stack=1,
        seq_length=1,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=None,
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        filter_by_attribute=None,
        load_next_obs=True,
        image_size=None,
        obs_info_keys=None,
    ):
        super(SequenceDataset, self).__init__()

        self.hdf5_path = os.path.expanduser(hdf5_path)
        self.hdf5_use_swmr = hdf5_use_swmr
        self.hdf5_cache_mode = hdf5_cache_mode
        self._hdf5_file = None

        # get all keys that needs to be fetched
        self.obs_keys = tuple(obs_keys)
        self.dataset_keys = tuple(dataset_keys)
        self.obs_info_keys = None
        if obs_info_keys is not None:
            self.obs_info_keys = obs_info_keys

        self.n_frame_stack = frame_stack
        assert self.n_frame_stack >= 1

        self.seq_length = seq_length
        assert self.seq_length >= 1

        self.image_size = image_size

        self.pad_seq_length = pad_seq_length
        self.pad_frame_stack = pad_frame_stack
        self.get_pad_mask = get_pad_mask

        self.filter_by_attribute = filter_by_attribute

        self.goal_mode = goal_mode
        self.load_next_obs = load_next_obs

        self.load_demo_info(filter_by_attribute=self.filter_by_attribute)

        self.counter = 0

        # # maybe prepare for observation normalization
        # self.obs_normalization_stats = None
        # if self.hdf5_normalize_obs:
        #     self.obs_normalization_stats = self.normalize_obs()

        # # maybe store dataset in memory for fast access
        # if self.hdf5_cache_mode in ["all", "all_nogetitem", "low_dim"]:
        #     # obs_keys_in_memory = self.obs_keys
        #     # if self.hdf5_cache_mode == "low_dim":
        #     #     # only store low-dim observations
        #     #     obs_keys_in_memory = []
        #     #     for k in self.obs_keys:
        #     #         if ObsUtils.key_is_obs_modality(k, "low_dim"):
        #     #             obs_keys_in_memory.append(k)
        #     # self.obs_keys_in_memory = obs_keys_in_memory

        #     self.hdf5_cache = self.load_dataset_in_memory(
        #         demo_list=self.demos,
        #         hdf5_file=self.hdf5_file,
        #         obs_keys=self.obs_keys_in_memory,
        #         dataset_keys=self.dataset_keys,
        #         load_next_obs=self.load_next_obs
        #     )

        #     if self.hdf5_cache_mode == "all":
        #         # cache getitem calls for even more speedup. We don't do this for
        #         # "low-dim" since image observations require calls to getitem anyways.
        #         print("SequenceDataset: caching get_item calls...")
        #         self.getitem_cache = [self.get_item(i) for i in LogUtils.custom_tqdm(range(len(self)))]

        #         # don't need the previous cache anymore
        #         del self.hdf5_cache
        #         self.hdf5_cache = None
        # else:
        #     self.hdf5_cache = None

    @property
    def hdf5_file(self):
        """
        This property allows for a lazy hdf5 file open.
        """
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=self.hdf5_use_swmr, libver='latest')
        return self._hdf5_file
    
    def load_demo_info(self, filter_by_attribute=None, demos=None):
        """
        Args:
            filter_by_attribute (str): if provided, use the provided filter key
                to select a subset of demonstration trajectories to load

            demos (list): list of demonstration keys to load from the hdf5 file. If 
                omitted, all demos in the file (or under the @filter_by_attribute 
                filter key) are used.
        """
        # filter demo trajectory by mask
        if demos is not None:
            self.demos = demos
        elif filter_by_attribute is not None:
            self.demos = [elem.decode("utf-8") for elem in np.array(self.hdf5_file["mask/{}".format(filter_by_attribute)][:])]
        else:
            self.demos = list(self.hdf5_file['data'].keys())

        # sort demo keys
        inds = np.argsort([int(elem[8:]) for elem in self.demos])
        self.demos = [self.demos[i] for i in inds]

        self.n_demos = len(self.demos)

        # keep internal index maps to know which transitions belong to which demos
        self._index_to_demo_id = dict()  # maps every index to a demo id
        self._demo_id_to_start_indices = dict()  # gives start index per demo id
        self._demo_id_to_demo_length = dict()

        # determine index mapping
        self.total_num_sequences = 0
        for ep in self.demos:
            # demo_length = self.hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            demo_length = len(self.hdf5_file["data/{}".format(ep)]['actions']['actions'])
            self._demo_id_to_start_indices[ep] = self.total_num_sequences
            self._demo_id_to_demo_length[ep] = demo_length

            num_sequences = demo_length
            # determine actual number of sequences taking into account whether to pad for frame_stack and seq_length
            if not self.pad_frame_stack:
                num_sequences -= (self.n_frame_stack - 1)
            if not self.pad_seq_length:
                num_sequences -= (self.seq_length - 1)

            # # Added by arpit
            # num_sequences = 1

            if self.pad_seq_length:
                assert demo_length >= 1  # sequence needs to have at least one sample
                num_sequences = max(num_sequences, 1)
            else:
                if num_sequences < 1:
                    print(f"Sequence {ep} can't be used with this sequence length")
                    #assert num_sequences >= 1  # assume demo_length >= (self.n_frame_stack - 1 + self.seq_length)

            # print("num_sequences: ", num_sequences)
            for _ in range(num_sequences):
                self._index_to_demo_id[self.total_num_sequences] = ep
                self.total_num_sequences += 1

    def __repr__(self):
        """
        Pretty print the class and important attributes on a call to `print`.
        """
        msg = str(self.__class__.__name__)
        msg += " (\n\tpath={}\n\tobs_keys={}\n\tseq_length={}\n\tfilter_key={}\n\tframe_stack={}\n"
        msg += "\tpad_seq_length={}\n\tpad_frame_stack={}\n\tgoal_mode={}\n"
        msg += "\tcache_mode={}\n"
        msg += "\tnum_demos={}\n\tnum_sequences={}\n)"
        filter_key_str = self.filter_by_attribute if self.filter_by_attribute is not None else "none"
        goal_mode_str = self.goal_mode if self.goal_mode is not None else "none"
        cache_mode_str = self.hdf5_cache_mode if self.hdf5_cache_mode is not None else "none"
        msg = msg.format(self.hdf5_path, self.obs_keys, self.seq_length, filter_key_str, self.n_frame_stack,
                         self.pad_seq_length, self.pad_frame_stack, goal_mode_str, cache_mode_str,
                         self.n_demos, self.total_num_sequences)
        return msg
        
    def __len__(self):
        """
        Ensure that the torch dataloader will do a complete pass through all sequences in 
        the dataset before starting a new iteration.
        """
        return self.total_num_sequences
    
    def extract_observations_info_from_hdf5(self, obs_info_strings, obs_info_shapes):
        # Reconstruct original structure
        idx = 0
        reconstructed_data = []
        for shape in obs_info_shapes:
            sublist = []
            for _ in range(shape):
                sublist.append(list(map(lambda x: x.decode('utf-8'), obs_info_strings[idx:idx+2])))
                idx += 2
            reconstructed_data.append(sublist)

        reconstructed_data = np.array(reconstructed_data, dtype=object)
        # for i in range(len(reconstructed_data)):
        #     print(i, np.array(reconstructed_data[i]).shape)
        #     if i == 0:
        #         print(reconstructed_data)
        return reconstructed_data
    
    def obtain_gripper_obj_seg(self, img, img_info):
        # img = f[f'data/{k}/observations/seg_instance_id'][0]
        # img_info = np.array(f[f'data/{k}/observations_info']['seg_instance_id']).astype(str)[0]
        parts_of_concern = [  
            '/World/robot0/gripper_right_link/visuals',
            '/World/robot0/gripper_right_right_finger_link/visuals',
            '/World/robot0/gripper_right_left_finger_link/visuals',
            '/World/coffee_table_fqluyq_0/base_link/visuals',
            '/World/box/base_link/visuals',
            '/World/plate/base_link/visuals',
            '/World/apple/base_link/visuals',
            '/World/table/base_link/visuals'
        ]
        ids_of_concern = []
        for row in img_info:
            key, val = row[0], row[1]
            # print("val: ", val)
            if val in parts_of_concern:
                ids_of_concern.append(int(key))
        
        # # print("ids_of_concern: ", ids_of_concern)
        # new_img = img.copy()
        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):
        #         if int(img[i][j]) not in ids_of_concern:
        #             new_img[i][j] = 0
                
        # new_img = img.copy()
        # new_img[np.argwhere(img not in ids_of_concern)] = 0
        
        # ids_of_concern = set(ids_of_concern)  # Convert to set for faster lookups, if not already
        # print("ids_of_concern: ", ids_of_concern)
        # Create a boolean mask where True indicates values in `ids_of_concern`
        img = img.astype(int)
        ids_of_concern = np.array(ids_of_concern).astype(int)
        mask = np.isin(img, ids_of_concern)
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(img)
        # ax[1].imshow(mask)
        # plt.show()
        # Create a copy of the original image
        new_img = img.copy()
        # Set all elements not in `ids_of_concern` to 0
        new_img[~mask] = 0
        
        return new_img
    
    def get_seg_instance_id_info(self, ep):
        # Basically dealing with HDF5 limitation: handling inconsistent length arrays in observations_info/seg_instance_id
        if 'seg_instance_id_strings' in self.hdf5_file[f'data/{ep}/observations_info'].keys():
            seg_instance_id_strings = np.array(self.hdf5_file["data/{}/observations_info/seg_instance_id_strings".format(ep)])
            seg_instance_id_shapes = np.array(self.hdf5_file["data/{}/observations_info/seg_instance_id_shapes".format(ep)])
            seg_instance_id = self.extract_observations_info_from_hdf5(obs_info_strings=seg_instance_id_strings, 
                                                                        obs_info_shapes=seg_instance_id_shapes)
        else:
            hd5key = "data/{}/observations_info/seg_instance_id".format(ep)
            seg_instance_id = self.hdf5_file[hd5key]
        return seg_instance_id
    
    def get_dataset_for_ep(self, ep, key):
        """
        Helper utility to get a dataset for a specific demonstration.
        Takes into account whether the dataset has been loaded into memory.
        """

        # check if this key should be in memory
        key_should_be_in_memory = (self.hdf5_cache_mode in ["all", "all_nogetitem", "low_dim"])
        if key_should_be_in_memory:
            # if key is an observation, it may not be in memory
            if '/' in key:
                key1, key2 = key.split('/')
                assert(key1 in ['obs', 'next_obs'])
                if key2 not in self.obs_keys_in_memory:
                    key_should_be_in_memory = False
        if key_should_be_in_memory:
            # read cache
            if '/' in key:
                key1, key2 = key.split('/')
                assert(key1 in ['obs', 'next_obs'])
                ret = self.hdf5_cache[ep][key1][key2]
            else:
                ret = self.hdf5_cache[ep][key]
        else:
            # read from file
            # print("keyyyy: ", key)
            if key == 'actions':
                hd5key = "data/{}/{}/{}".format(ep, key, key)
            elif key == 'obs/rgb':
                # hd5key = "data/{}/observations/gripper_obj_seg".format(ep) 
                gripper_obj_seg_imgs = []
                seg_instance_id_img = self.hdf5_file["data/{}/observations/seg_instance_id".format(ep) ]
                seg_instance_id_info = self.get_seg_instance_id_info(ep)
                for seq_num in range(seg_instance_id_img.shape[0]):
                    gripper_obj_seg_img = self.obtain_gripper_obj_seg(seg_instance_id_img[seq_num], seg_instance_id_info[seq_num])
                    gripper_obj_seg_imgs.append(gripper_obj_seg_img)
                return np.array(gripper_obj_seg_imgs)
            elif key == 'grasped':
                hd5key = "data/{}/extras/grasps".format(ep)
            elif key == 'obs/seg_instance_id_info':
                return self.get_seg_instance_id_info(ep)
            
            ret = self.hdf5_file[hd5key]
        return ret
    
    def get_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1):
        """
        Extract a (sub)sequence of data items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range

        Returns:
            a dictionary of extracted items.
        """
        assert num_frames_to_stack >= 0
        assert seq_length >= 1

        demo_length = self._demo_id_to_demo_length[demo_id]
        assert index_in_demo < demo_length

        # determine begin and end of sequence (This is what gives us multiple sequences per episode [IMPORTANT])
        seq_begin_index = max(0, index_in_demo - num_frames_to_stack)
        seq_end_index = min(demo_length, index_in_demo + seq_length)

        # determine sequence padding
        seq_begin_pad = max(0, num_frames_to_stack - index_in_demo)  # pad for frame stacking
        seq_end_pad = max(0, index_in_demo + seq_length - demo_length)  # pad for sequence length
        # print("seq_end_pad: ", seq_end_pad)

        # make sure we are not padding if specified.
        if not self.pad_frame_stack:
            assert seq_begin_pad == 0
        if not self.pad_seq_length:
            assert seq_end_pad == 0

        # fetch observation from the dataset file
        seq = dict()
        for k in keys:
            data = self.get_dataset_for_ep(demo_id, k)
            # seq[k] = data[seq_begin_index: seq_end_index].astype("float32")
            # Retain existing datatype
            seq[k] = data[seq_begin_index: seq_end_index]
            # change grasped from bool to float
            if k == 'grasped':
                seq[k] = seq[k].astype(float)

            # print("seq[k]: ", seq[k].shape)

        seq = pad_sequence(seq, padding=(seq_begin_pad, seq_end_pad), pad_same=True)
        # if 'actions' not in keys:        
        #     print("seq: ", seq[k][-1, :])
        pad_mask = np.array([0] * seq_begin_pad + [1] * (seq_end_index - seq_begin_index) + [0] * seq_end_pad)
        pad_mask = pad_mask[:, None].astype(np.bool)

        return seq, pad_mask
    
    def get_dataset_sequence_from_demo(self, demo_id, index_in_demo, keys, seq_length=1):
        """
        Extract a (sub)sequence of dataset items from a demo given the @keys of the items (e.g., states, actions).
        
        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range

        Returns:
            a dictionary of extracted items.
        """
        data, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=keys,
            num_frames_to_stack=0,  # don't frame stack for meta keys
            seq_length=seq_length,
        )
        if self.get_pad_mask:
            data["pad_mask"] = pad_mask
        return data
    
    def get_obs_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1, prefix="obs", obs_info_keys=None):
        """
        Extract a (sub)sequence of observation items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range
            prefix (str): one of "obs", "next_obs"

        Returns:
            a dictionary of extracted items.
        """
        obs, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=tuple('{}/{}'.format(prefix, k) for k in keys),
            num_frames_to_stack=num_frames_to_stack,
            seq_length=seq_length,
        )
        
        if obs_info_keys is not None:
            obs_info, pad_mask = self.get_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=tuple('{}/{}'.format(prefix, k) for k in obs_info_keys),
                num_frames_to_stack=num_frames_to_stack,
                seq_length=seq_length,
            )
        
        obs = {k.split('/')[1]: obs[k] for k in obs}  # strip the prefix
        obs_info = {k.split('/')[1]: obs_info[k] for k in obs_info}  # strip the prefix

        if self.get_pad_mask:
            obs["pad_mask"] = pad_mask
            obs_info["pad_mask"] = pad_mask

        # prepare image observations from dataset
        for k in obs:
            # print("1obs[k].shape: ", obs[k].shape)  

            # uncomment later
            # obs[k] = obs[k][:, :, :, :3]
            # obs[k] = np.transpose(obs[k], (0, 3, 1, 2))

            # # Remove later: Categorical to one-hot 
            # seq_len, h, w = obs[k].shape[0], obs[k].shape[1], obs[k].shape[2]
            # one_hot_encoded_image = np.zeros((seq_len, h, w, 20), dtype=int)
            # for s in range(seq_len):
            #     for i in range(h):
            #         for j in range(w):
            #             label = obs[k][s, i, j]
            #             one_hot_encoded_image[s, i, j, label] = 1 

            # OPTION 1: Randomizing the classes of all objects
            # # Make all classes of the gripper (the fingers, body etc.) as one class
            # temp_obs = obs[k].copy()
            # obs_info = obs_info['seg_instance_id_info']
            # # print("obs_info: ", obs_info, len(obs_info))
            # # if len(obs_info[0]) > 0:
            # # find the class of gripper_right_link
            # # TODO: find all classes in this seq len and make the initial class of gripper_right_link different from these 
            # unique_class_arr = np.unique(obs[k])
            # # print("unique_class_arr: ", unique_class_arr)
            # all_classes = set(range(20))
            # # Remove the elements in A from the set of all integers
            # available_classes = list(all_classes - set(unique_class_arr))
            # # Choose a random integer from the available integers
            # gripper_right_link_class = np.random.choice(available_classes)
            # # print("gripper_right_link_class: ", gripper_right_link_class)
            # # gripper_right_link_class = 19
            
            # gripper_right_right_finger_link_class, gripper_right_left_finger_link_class = None, None  
            # obs_info_copy = obs_info.copy()
            # for seq_num in range(len(obs_info)):
            #     for i, cls in enumerate(obs_info[seq_num]):
            #         if cls[1] == '/World/robot0/gripper_right_link/visuals':
            #             gripper_right_link_class = int(cls[0])
            #         if cls[1] == '/World/robot0/gripper_right_right_finger_link/visuals':
            #             gripper_right_right_finger_link_class = int(cls[0])
            #         if cls[1] == '/World/robot0/gripper_right_left_finger_link/visuals':
            #             gripper_right_left_finger_link_class = int(cls[0])

            # # print("gripper, right, left fingers: ", gripper_right_link_class, gripper_right_right_finger_link_class, gripper_right_left_finger_link_class)
            # # print("obs[k] shape: ", type(obs[k]), obs[k].dtype)
            # if gripper_right_right_finger_link_class is not None:
            #     obs[k][obs[k] == gripper_right_right_finger_link_class] = gripper_right_link_class
            # if gripper_right_left_finger_link_class is not None:
            #     obs[k][obs[k] == gripper_right_left_finger_link_class] = gripper_right_link_class
                
            # for seq_num in range(len(obs_info)):
            #     for i, cls in enumerate(obs_info[seq_num]):
            #         if cls[1] == '/World/robot0/gripper_right_link/visuals':
            #             obs_info_copy[seq_num][i][0] = gripper_right_link_class
            #         if cls[1] == '/World/robot0/gripper_right_right_finger_link/visuals':
            #             obs_info_copy[seq_num][i][0] = gripper_right_link_class
            #         if cls[1] == '/World/robot0/gripper_right_left_finger_link/visuals':
            #             obs_info_copy[seq_num][i][0] = gripper_right_link_class
            
            # temp2 = obs[k].copy()
            # obs_copy = obs[k].copy()

            # # randomize the classes
            # old_to_new_class_dict = {i: i for i in range(20)}
            # all_classes = np.arange(1,20)
            # for unique_class in unique_class_arr:
            #     # keeping background always 0
            #     if unique_class == 0:
            #         continue
            #     new_cls = np.random.choice(all_classes)
            #     # print("before: ", all_classes)
            #     # print("current class: ", unique_class)
            #     # print("new_cls: ", new_cls)
            #     all_classes = np.delete(all_classes, np.argwhere(all_classes==new_cls))
            #     # print("after: ", all_classes)
            #     obs_copy[obs[k] == unique_class] = new_cls
            #     old_to_new_class_dict[unique_class] = new_cls
            # obs[k] = obs_copy

            # OPTION 2: Make fixed classes for each object (useful when need different weights for different classes in the cross entropy loss)
            obs_info_with_new_cls_padded = []
            obs_copy = obs[k].copy()
            obs_info = obs_info['seg_instance_id_info']
            old_to_new_class_dict = {i: i for i in range(20)}
            fixed_classes = {
                '/World/robot0/gripper_right_link/visuals': 1,
                '/World/robot0/gripper_right_right_finger_link/visuals': 1,
                '/World/robot0/gripper_right_left_finger_link/visuals': 1,
                # '/World/coffee_table_fqluyq_0/base_link/visuals',
                # '/World/box/base_link/visuals',
                '/World/table/base_link/visuals': 2,
                '/World/apple/base_link/visuals': 3,
                '/World/plate/base_link/visuals': 4,
            }
            # print("obs_info shape: ", obs_info.shape)
            for class_name, target_class_value in fixed_classes.items():
                found_class = False
                class_value_in_current_episode = -1
                # class_value_in_current_frame = obs_info[0, obs_info[:, 0] == class_name, 1]
                for seq_num in range(len(obs_info)):
                    for i, cls in enumerate(obs_info[seq_num]):
                        # print("cls[1], class_name: ", cls[1], class_name)
                        if cls[1] == class_name:
                            # print("cls[1], class_name: ", cls[1], class_name)
                            class_value_in_current_episode = int(cls[0])
                            found_class = True
                            break
                    if found_class:
                        break

                if class_value_in_current_episode != -1:
                    # print("class_name, current class value: ", class_name, class_value_in_current_episode)
                    obs_copy[obs[k] == class_value_in_current_episode] = target_class_value
            # fig, ax = plt.subplots(1,2)
            # ax[0].imshow(obs[k][0])
            # ax[1].imshow(obs_copy[0])
            # plt.show()
            obs[k] = obs_copy

            # print("old_to_new_class_dict:",old_to_new_class_dict)
            
            # obs_info_with_new_cls = obs_info_copy.copy()
            # for seq_num in range(len(obs_info_copy)):
            #     for i, cls in enumerate(obs_info_copy[seq_num]):
            #         if int(cls[0]) in old_to_new_class_dict.keys():
            #             obs_info_with_new_cls[seq_num][i][0] = str(old_to_new_class_dict[int(cls[0])])

            # # Hack for hdf5: make obs_info a constant sized array (seq_len, 20, 2)
            # obs_info_with_new_cls_padded = []
            # target_length=20
            # for seq in obs_info_with_new_cls:
            #     # Calculate the number of padding elements needed
            #     padding_needed = target_length - len(seq)
            #     # print("padding_needed: ", padding_needed)
                
            #     # Create the padding array with ['', ''] repeated for the padding needed
            #     padding = [['', '']] * padding_needed
            #     # print("dtypes: ", np.array(seq).dtype, np.array(padding).dtype, type(seq[0][0]))
                
            #     # Append the padding to the sequence
            #     padded_seq = np.vstack([seq, padding])
            #     # print("padded_seq: ", padded_seq.shape)
                
            #     obs_info_with_new_cls_padded.append(padded_seq)
            
            # obs_info_with_new_cls_padded = np.array(obs_info_with_new_cls_padded)
            # # print("obs_info_with_new_cls_padded: ", obs_info_with_new_cls_padded)
            # # print("obs_info_with_new_cls: ", obs_info_with_new_cls)
            # # input()
            # # import matplotlib.pyplot as plt
            # # fig, ax = plt.subplots(1,3)
            # # ax[0].imshow(temp_obs[0])
            # # ax[1].imshow(temp2[0])
            # # ax[2].imshow(obs[k][0])
            # # plt.show()

            # vectorized: Very cool vectorization!!
            seq_len, h, w = obs[k].shape[0], obs[k].shape[1], obs[k].shape[2]
            one_hot_encoded_image = np.zeros((seq_len, h, w, 20), dtype=int)
            one_hot_encoded_image[np.arange(seq_len)[:, None, None], np.arange(h)[None, :, None], np.arange(w)[None, None, :], obs[k]] = 1


            # One-hot to categorical
            # categorical_array = np.argmax(one_hot_encoded_image, axis=-1)
            # # another way 
            # indices = np.where(one_hot_encoded_image == 1)
            # categorical_array = indices[-1].reshape(one_hot_encoded_image.shape[:-1])
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1,2)
            # ax[0].imshow(obs[k][0])
            # ax[1].imshow(categorical_array[0])
            # plt.show()
                            
            obs[k] = np.transpose(one_hot_encoded_image, (0, 3, 1, 2))

            # # remove later
            # gt = one_hot_encoded_image
            # for b in range(bs):
            #     for i in range(h):
            #         for j in range(w):
            #             gt_label = np.where(gt[b, i, j] == 1)
            #             if len(gt_label[0]) == 0:
            #                 print("gt[b, s, i, j]: ",gt[b, i, j])
            #                 print("wowwwwwwwwwwwww")

            # print("2obs[k].shape: ", obs[k].shape)   
        # print("obs: ", np.array(obs['agentview_shift_2_image']).shape, type(obs['agentview_shift_2_image']), type(obs['agentview_shift_2_image'][0][0][0][0]))
        # print("===================================================")
        # temp = ObsUtils.process_obs_dict(obs)
        # print("obs: ", np.array(temp['agentview_shift_2_image']).shape, type(obs['agentview_shift_2_image']), type(obs['agentview_shift_2_image'][0][0][0][0]))

        return obs, obs_info_with_new_cls_padded
    
    def __getitem__(self, index):
        """
        Fetch dataset sequence @index (inferred through internal index map), using the getitem_cache if available.
        """
        # self.counter += 1
        # print("self.counter: ", self.counter, index)
        if self.hdf5_cache_mode == "all":
            return self.getitem_cache[index]
        return self.get_item(index)
    
    def get_item(self, index):
        """
        Main implementation of getitem when not using cache.
        """
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        
        # # remove later
        # demo_index_offset = 5

        index_in_demo = index - demo_start_index + demo_index_offset
        # print("index, index_in_demo: ", index, index_in_demo)

        # # end at offset index if not padding for seq length
        # demo_length_offset = 0 if self.pad_seq_length else (self.seq_length - 1)
        # end_index_in_demo = demo_length - demo_length_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.dataset_keys,
            seq_length=self.seq_length
        )

        meta["obs"], obs_info = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=self.seq_length,
            prefix="obs",
            obs_info_keys=self.obs_info_keys
        )

        # print("meta_obs: ", meta.keys(), type(meta['obs']['rgb']), type(meta['actions']), type(meta['grasped']))
        # print("obs_info: ", np.array(obs_info).shape, type(obs_info[0][0][0]))
        # meta['obs_info'] = obs_info.tolist()

        # breakpoint()
        # print("11meta_obs: ", meta['obs']['rgb'].shape, meta['obs']['rgb'][0,:,0,0])
        #TODO: commented out resize but bring it back
        meta["obs"] = self.resize_image_observations(meta["obs"], max=255)

        # # remove later
        # print("==============================================================================")
        # gt = meta["obs"]['rgb'].transpose(0,2,3,1)
        # bs, h, w = gt.shape[0], gt.shape[1], gt.shape[2]
        # for b in range(bs):
        #     for i in range(h):
        #         for j in range(w):
        #             gt_label = np.where(gt[b, i, j] == 1)
        #             if len(gt_label[0]) == 0:
        #                 print("gt[b, s, i, j]: ",gt[b, i, j])
        #                 print("wowwwwwwwwwwwww")
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        # print("22meta_obs: ", meta['obs']['rgb'].shape, meta['obs']['rgb'][0,:,0,0])
        return meta
    
    def resize_image_observations(self, obs, format='chw', max=1):
        # really gross, refactor
        def resize_tensor(t, dims, interpolation_mode):
            h, w = dims
            if t.shape[-2] == h and t.shape[-1] == w:
                return t
            else:
                # uses Bilinear interpolation by default, use antialiasing
                if interpolation_mode == InterpolationMode.BILINEAR:
                    antialias=True
                else:
                    antialias = False
                t = Resize(dims, interpolation=interpolation_mode, antialias=antialias)(t)
                return t

        if self.image_size is None:
            # no resize specified
            return obs
        else:
            for k, v in obs.items():
                if v.shape[-2] == self.image_size[0] and v.shape[-1] == self.image_size[1]:
                    return obs
                interpolation_mode = InterpolationMode.BILINEAR
                if 'seg' in k: # for segmentations, use Interpolation mode nearest to make sure things are still integers
                    interpolation_mode = InterpolationMode.NEAREST

                # Added by Arpit. Forcing this interpolation mode as our obs is a segmentation
                interpolation_mode = InterpolationMode.NEAREST

                # if we have a video of any type, resize it
                if 'seg' in k or 'depth' in k:
                    if len(v.shape) == 3:
                        if format == 'chw':
                            v = v[:, None]
                        else:
                            v = v[..., None]
                    clamp_max = v.max()
                else:
                    clamp_max = max

                if len(v.shape) > 3:
                    if format == 'chw':
                        obs[k] = resize_tensor(torch.tensor(v), self.image_size, interpolation_mode).numpy()
                    else:
                        obs[k] = np.moveaxis(resize_tensor(torch.tensor(np.moveaxis(v, 3, 1)), self.image_size, interpolation_mode).numpy(), 1, 3)
                        # obs[k] = np.squeeze(obs[k])
                    obs[k] = np.clip(obs[k], 0, clamp_max)  # prevent interpolation numerical issues
            return obs