import torch
import numpy as np

from robomimic.utils.dataset import SequenceDataset
import robomimic.utils.obs_utils as ObsUtils
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Resize


def get_image_name(cam):
    if not cam:
        return 'image'  # for the real world datasets
    else:
        return f'{cam}_image'


def get_data_loader(dataset_paths, batch_size, video_len, video_dims, phase, depth, normal, view, cache_mode='lowdim',
                    seg=True, only_depth=False, only_state=False):
    """
    Get a data loader to sample batches of data.
    """
    imageview_name = get_image_name(view)

    ObsUtils.initialize_obs_utils_with_obs_specs({
        "obs": {
            "rgb": [imageview_name, f'{view}_normal'],
            "depth": [f"{view}_depth"],
            # "scan": [f"{view}_segmentation_instance"]
            "scan": [f"{view}_seg"],
            # "normal": [f"{view}_normal"]
            "low_dim": ["object", "robot0_eef_pos", "robot0_eef_quat"],
        }
    })

    all_datasets = []
    for i, dataset_path in enumerate(dataset_paths):
        # obs_keys = (f"{view}_image", f"{view}_segmentation_instance")
        # obs_keys = ("object", "robot0_eef_pos", "robot0_eef_quat")
        obs_keys = tuple()
        if not only_depth and not only_state:
            obs_keys = obs_keys + (imageview_name,)
        if depth or only_depth:
            obs_keys = obs_keys + (f"{view}_depth",)
        if seg:
            obs_keys = obs_keys + (f"{view}_seg",)
        if normal:
            obs_keys = obs_keys + (f"{view}_normal",)

        dataset = SequenceDataset(
            hdf5_path=dataset_path,
            obs_keys=obs_keys,  # observations we want to appear in batches
            dataset_keys=(  # can optionally specify more keys here if they should appear in batches
                "actions",
                "rewards",
                "dones",
            ),
            load_next_obs=False,
            frame_stack=1,
            seq_length=video_len,  # length-10 temporal sequences
            pad_frame_stack=True,
            pad_seq_length=False,  # pad last obs per trajectory to ensure all sequences are sampled
            get_pad_mask=False,
            goal_mode=None,
            hdf5_cache_mode=cache_mode,  # cache dataset in memory to avoid repeated file i/o
            hdf5_use_swmr=True,
            hdf5_normalize_obs=False,
            filter_by_attribute=phase,  # filter either train or validation data
            image_size=video_dims,
        )
        all_datasets.append(dataset)
        print(f"\n============= Created Dataset {i + 1} out of {len(dataset_paths)} =============")
        print(dataset)
        print("")
    dataset = ConcatDataset(all_datasets)
    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,  # no custom sampling logic (uniform sampling)
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,  # don't provide last batch in dataset pass if it's less than 100 in size
    )
    return data_loader


def load_dataset_robomimic_torch(dataset_path, batch_size, video_len, video_dims, phase, depth, normal, view='agentview',
                                 cache_mode='low_dim', seg=True, only_depth=False, only_state=False, augmentation=None):
    assert phase in ['train', 'valid'], f'Phase is not one of the acceptable values! Got {phase}'

    loader = get_data_loader(dataset_path, batch_size, video_len, video_dims, phase, depth, normal, view, cache_mode, seg,
                             only_depth, only_state)

    def prepare_data(xs):
        if only_state:
            data_dict = {
                'video': torch.cat([xs['obs']['robot0_eef_pos'],
                                    xs['obs']['robot0_eef_quat'],
                                    xs['obs']['object']], dim=-1),
                'actions': xs['actions'],
            }
        elif only_depth:
            # take depth video as the actual video
            data_dict = {
                'video': xs['obs'][f'{view}_depth'],
                'actions': xs['actions'],
            }
        else:
            data_dict = {
                'video': xs['obs'][get_image_name(view)],
                'actions': xs['actions'],
            }

        if augmentation:
            data_dict["video"] = augmentation(data_dict["video"])

        if f'{view}_seg' in xs['obs']:
            data_dict['segmentation'] = xs['obs'][f'{view}_seg']
            # from perceptual_metrics.mpc.utils import save_np_img
            # import ipdb; ipdb.set_trace()
            # for i in range(10):
            #     save_np_img(np.tile(((data_dict['segmentation'][0, i, 0] == i) * 60).cpu().numpy()[..., None], (1, 1, 3)).astype(np.uint8), f'seg_{i}')
            # zero out the parts of the segmentation which are not assigned label corresponding to object of interest
            # set the object label components to 1
            object_seg_indxs = [0, 1, 2, 3]  # Seg index is 0 on the iGibson data, and 1 on Mujoco data
            arm_seg_indxs = [4, 5, 6]  # Seg index is 0 on the iGibson data, and 1 on Mujoco data
            seg_image = torch.zeros_like(data_dict['segmentation'])
            for object_seg_index in object_seg_indxs:
                seg_image[data_dict['segmentation'] == object_seg_index] = 1
            for arm_seg_index in arm_seg_indxs:
                seg_image[data_dict['segmentation'] == arm_seg_index] = 2
            not_either_mask = ~(
                        (seg_image == 1) | (seg_image == 2))
            seg_image[not_either_mask] = 0
            data_dict['segmentation'] = seg_image
        else:
            data_dict['segmentation'] = None
        if depth and not only_depth:
            data_dict['depth_video'] = xs['obs'][f'{view}_depth']
        if normal:
            data_dict['normal'] = xs['obs'][f'{view}_normal']

        if "video" in data_dict:
            # Normalize to [0, 1]
            data_dict["video"] = data_dict["video"] / 255.

        return data_dict

    return loader, prepare_data


if __name__ == '__main__':
    dataset_path = '/viscam/u/stian/perceptual-metrics/robomimic/datasets/lift/mg/image_and_depth.hdf5'
    dataset_path = '/viscam/u/stian/perceptual-metrics/robosuite/robosuite/models/assets/demonstrations/1644373519_3708425/1644373519_3708425_igibson_obs.hdf5'
    dataset_path = '/viscam/u/stian/perceptual-metrics/robosuite/robosuite/models/assets/policy_rollouts/pushcenter_osc_position_eval/igibson_obs.hdf5'

    dl, p = load_dataset_robomimic_torch([dataset_path], 16, 10, 'train', depth=False, view='agentview_shift_2')
    batch = next(iter(dl))
    p(batch)

    # dataloader = get_data_loader(dataset_path, 1, 10, 'train')
    # batch = next(iter(dataloader))