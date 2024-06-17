import torch
import numpy as np

# from robomimic.utils.dataset import SequenceDataset
from fitvid.data.og_dataset import SequenceDataset
import robomimic.utils.obs_utils as ObsUtils
from torch.utils.data import DataLoader, ConcatDataset

# Hacky import, in newer versions of pytorch this is easier to import
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import Resize


def get_image_name(cam):
    if cam == 'rgb':
        return cam
    if not cam:
        return "image"  # for the real world datasets
    else:
        return f"{cam}_image"


def get_data_loader(
    dataset_paths,
    batch_size,
    video_len,
    video_dims,
    phase,
    depth,
    normal,
    view,
    collate_fn,
    cache_mode="lowdim",
    seg=True,
    only_depth=False,
    only_state=False,
    shuffle=True
):
    """
    Get a data loader to sample batches of data.
    """
    imageview_name = get_image_name(view)

    # ObsUtils.initialize_obs_utils_with_obs_specs(
    #     {
    #         "obs": {
    #             "rgb": [imageview_name, f"{view}_normal"],
    #             "depth": [f"{view}_depth"],
    #             # "scan": [f"{view}_segmentation_instance"]
    #             "scan": [f"{view}_seg"],
    #             # "normal": [f"{view}_normal"]
    #             "low_dim": ["object", "robot0_eef_pos", "robot0_eef_quat"],
    #         }
    #     }
    # )

    obs_keys = tuple()
    if not only_depth and not only_state:
        obs_keys = obs_keys + (imageview_name,)
    if depth or only_depth:
        obs_keys = obs_keys + (f"{view}_depth",)
    if seg:
        obs_keys = obs_keys + (f"{view}_seg",)
    if normal:
        obs_keys = obs_keys + (f"{view}_normal",)

    print("obs_keys: ", obs_keys)

    all_datasets = []

    for i, dataset_path in enumerate(dataset_paths):
        # obs_keys = (f"{view}_image", f"{view}_segmentation_instance")
        # obs_keys = ("object", "robot0_eef_pos", "robot0_eef_quat")
        # print("dataset_path: ", dataset_path)
        # print("cache_mode: ", cache_mode)

        # dataset = SequenceDataset(
        #     hdf5_path=dataset_path,
        #     obs_keys=obs_keys,  # observations we want to appear in batches
        #     dataset_keys=(  # can optionally specify more keys here if they should appear in batches
        #         "actions",
        #         "rewards",
        #         "dones",
        #     ),
        #     load_next_obs=False,
        #     frame_stack=1,
        #     seq_length=video_len,  # length-10 temporal sequences
        #     pad_frame_stack=True,
        #     pad_seq_length=False,  # pad last obs per trajectory to ensure all sequences are sampled
        #     get_pad_mask=False,
        #     goal_mode=None,
        #     hdf5_cache_mode=cache_mode,  # cache dataset in memory to avoid repeated file i/o
        #     hdf5_use_swmr=True,
        #     hdf5_normalize_obs=False,
        #     filter_by_attribute=phase,  # filter either train or validation data
        #     image_size=video_dims,
        # )
        # # Added by Arpit. Loading og dataset
        dataset = SequenceDataset(
            hdf5_path=dataset_path,
            obs_keys=obs_keys,  # observations we want to appear in batches
            dataset_keys=(  # can optionally specify more keys here if they should appear in batches
                "actions",
            ),
            load_next_obs=False,
            frame_stack=1,
            seq_length=video_len,  # length-10 temporal sequences
            pad_frame_stack=False,
            pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
            get_pad_mask=False,
            goal_mode=None,
            hdf5_cache_mode=cache_mode,  # cache dataset in memory to avoid repeated file i/o
            hdf5_use_swmr=True,
            hdf5_normalize_obs=False,
            filter_by_attribute=phase,  # filter either train or validation data
            image_size=video_dims,
        )

        # temp = dataset[78]
        # print("datasettttt: ", temp.keys())
        # print("dataset[actions]: ", temp['actions'].shape, type(temp['actions'][0][0]))
        # print("dataset[dones]: ", temp['dones'], type(temp['dones'][0]))
        # print("dataset[rewards]: ", temp['rewards'], type(temp['rewards'][0]))
        # print("dataset[obs]: ", temp['obs']['agentview_shift_2_image'].shape, temp['obs'].keys())
    
        all_datasets.append(dataset)
        print(
            f"\n============= Created Dataset {i + 1} out of {len(dataset_paths)} ============="
        )
        print(dataset)
        print("len(dataset): ", len(dataset))
        print("")
    # print("all_data")
    dataset = ConcatDataset(all_datasets)
    # print("type(dataset): ", type(dataset))
    print("batch_size: ", batch_size)
    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,  # no custom sampling logic (uniform sampling)
        batch_size=batch_size,
        shuffle=shuffle, 
        num_workers=4,
        drop_last=True,  # don't provide last batch in dataset pass if it's less than 100 in size
        collate_fn=collate_fn,
    )
    return data_loader


def load_dataset_robomimic_torch(
    dataset_path,
    batch_size,
    video_len,
    video_dims,
    phase,
    depth,
    normal,
    view="agentview",
    cache_mode="low_dim", #change later to low_dim
    seg=True,
    only_depth=False,
    only_state=False,
    augmentation=None,
    postprocess_fn=None,
    shuffle=True
):
    assert phase in [
        "train",
        "valid",
        None,
    ], f"Phase is not one of the acceptable values! Got {phase}"

    def prepare_data(input_batch):
        # prepare_data is a custom collate function which not only batches the data from the dataset, but also
        # creates the output dictionaries which contain keys "video" and "actions"
        # print("input_batch: ", np.array(input_batch).shape)
        xs = default_collate(input_batch)
        # print("type(xs): ", type(xs), xs.keys())
        # print("xssssssss: ", xs['actions'].shape)
        
        # Added by Arpit to resolve the shape mismatch error (32, 145) x (142, 128)
        # print("actions: ", xs['actions'][1:3])
        # print("----", xs['actions'][:, :, :3].shape, xs['actions'][:, :, -1:].shape)
        # xs['actions'] = torch.cat((xs['actions'][:, :, :3], xs['actions'][:, :, -1:]), dim=2)
        # print("xssssssss: ", xs['actions'].shape)
        
        if only_state:
            data_dict = {
                "video": torch.cat(
                    [
                        xs["obs"]["robot0_eef_pos"],
                        xs["obs"]["robot0_eef_quat"],
                        xs["obs"]["object"],
                    ],
                    dim=-1,
                ),
                "actions": xs["actions"],
            }
        elif only_depth:
            # take depth video as the actual video
            data_dict = {
                "video": xs["obs"][f"{view}_depth"],
                "actions": xs["actions"],
            }
        else:
            data_dict = {
                "video": xs["obs"][get_image_name(view)],
                "actions": xs["actions"],
            }
        # data_dict["rewards"] = xs["rewards"]
        if augmentation:
            data_dict["video"] = augmentation(data_dict["video"])

        if f"{view}_seg" in xs["obs"]:
            data_dict["segmentation"] = xs["obs"][f"{view}_seg"]
            # from perceptual_metrics.mpc.utils import save_np_img
            # import ipdb; ipdb.set_trace()
            # for i in range(10):
            #     save_np_img(np.tile(((data_dict['segmentation'][0, i, 0] == i) * 60).cpu().numpy()[..., None], (1, 1, 3)).astype(np.uint8), f'seg_{i}')
            # zero out the parts of the segmentation which are not assigned label corresponding to object of interest
            # set the object label components to 1
            object_seg_indxs = [
                0,
                1,
                2,
                3,
            ]  # Seg index is 0 on the iGibson data, and 1 on Mujoco data
            arm_seg_indxs = [
                4,
                5,
                6,
            ]  # Seg index is 0 on the iGibson data, and 1 on Mujoco data
            seg_image = torch.zeros_like(data_dict["segmentation"])
            for object_seg_index in object_seg_indxs:
                seg_image[data_dict["segmentation"] == object_seg_index] = 1
            for arm_seg_index in arm_seg_indxs:
                seg_image[data_dict["segmentation"] == arm_seg_index] = 2
            not_either_mask = ~((seg_image == 1) | (seg_image == 2))
            seg_image[not_either_mask] = 0
            data_dict["segmentation"] = seg_image
        else:
            data_dict["segmentation"] = None
        if depth and not only_depth:
            data_dict["depth_video"] = xs["obs"][f"{view}_depth"]
        if normal:
            data_dict["normal"] = xs["obs"][f"{view}_normal"]

        if "video" in data_dict:
            # Normalize to [0, 1]
            data_dict["video"] = data_dict["video"] / 255.0

        if postprocess_fn:
            data_dict = postprocess_fn(data_dict)

        return data_dict

    loader = get_data_loader(
        dataset_path,
        batch_size,
        video_len,
        video_dims,
        phase,
        depth,
        normal,
        view,
        collate_fn=prepare_data,
        cache_mode=cache_mode,
        seg=seg,
        only_depth=only_depth,
        only_state=only_state,
        shuffle=shuffle
    )

    return loader


if __name__ == "__main__":
    # dataset_path = "/viscam/u/stian/perceptual-metrics/robomimic/datasets/lift/mg/image_and_depth.hdf5"
    # dataset_path = "/viscam/u/stian/perceptual-metrics/robosuite/robosuite/models/assets/demonstrations/1644373519_3708425/1644373519_3708425_igibson_obs.hdf5"
    # dataset_path = "/viscam/u/stian/perceptual-metrics/robosuite/robosuite/models/assets/policy_rollouts/pushcenter_osc_position_eval/igibson_obs.hdf5"

    # dataset_path = "/home/arpit/test_projects/vp2/vp2/robosuite_benchmark_tasks/5k_slice_rendered_256.hdf5"
    dataset_path = "/home/arpit/test_projects/OmniGibson/dynamics_model_data/succ.hdf5"

    # dl = load_dataset_robomimic_torch(
    #     [dataset_path], 32, 12, None, "train", depth=False, normal=False, view="agentview_shift_2", seg=False,
    # )
    dl = load_dataset_robomimic_torch(
        [dataset_path], 16, 8, None, phase=None, depth=False, normal=False, view="rgb", seg=False, cache_mode=None
    )
    print("Train data_loader len: ", len(dl))
    # dl_valid = load_dataset_robomimic_torch(
    #     [dataset_path], 32, 12, None, "valid", depth=False, normal=False, view="agentview_shift_2", seg=False,
    # )
    # print("Valid data_loader len: ", len(dl_valid))
    # batch = next(iter(dl))
    # print("batch: ", batch.keys())
    # print("batch[video]: ", batch['video'].shape)
    # print("batch[actions]: ", batch['actions'].shape)
    # p(batch)

    # dataloader = get_data_loader(dataset_path, 1, 10, 'train')
    # batch = next(iter(dataloader))
