import torch
import collections
import numpy as np
import h5py
import random

def resize_tensor(t, dims):
    h, w = dims
    if t.shape[-2] == h and t.shape[-1] == w:
        return t
    else:
        # uses Bilinear interpolation by default, use antialiasing
        orig_shape = tuple(t.shape[:-3])
        img_shape = tuple(t.shape[-3:])
        t = torch.reshape(t, (-1,) + img_shape)
        t = Resize(dims, antialias=True)(t)
        t = torch.reshape(t, orig_shape + (img_shape[0],) + tuple(dims))
        return t


def dict_to_cuda(d):
    # turn all pytorch tensors into cuda tensors, leave all other objects along
    d = {k: v.cuda() if torch.is_tensor(v) else v for k, v in d.items()}
    return d


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def recursive_dict_list_tuple_apply(x, type_func_dict):
    """
    Recursively apply functions to a nested dictionary or list or tuple, given a dictionary of 
    {data_type: function_to_apply}.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        type_func_dict (dict): a mapping from data types to the functions to be 
            applied for each data type.

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    assert(list not in type_func_dict)
    assert(tuple not in type_func_dict)
    assert(dict not in type_func_dict)

    if isinstance(x, (dict, collections.OrderedDict)):
        new_x = collections.OrderedDict() if isinstance(x, collections.OrderedDict) else dict()
        for k, v in x.items():
            new_x[k] = recursive_dict_list_tuple_apply(v, type_func_dict)
        return new_x
    elif isinstance(x, (list, tuple)):
        ret = [recursive_dict_list_tuple_apply(v, type_func_dict) for v in x]
        if isinstance(x, tuple):
            ret = tuple(ret)
        return ret
    else:
        for t, f in type_func_dict.items():
            if isinstance(x, t):
                return f(x)
        else:
            raise NotImplementedError(
                'Cannot handle data type %s' % str(type(x)))
        
def pad_sequence_single(seq, padding, batched=False, pad_same=True, pad_values=None):
    """
    Pad input tensor or array @seq in the time dimension (dimension 1).

    Args:
        seq (np.ndarray or torch.Tensor): sequence to be padded
        padding (tuple): begin and end padding, e.g. [1, 1] pads both begin and end of the sequence by 1
        batched (bool): if sequence has the batch dimension
        pad_same (bool): if pad by duplicating
        pad_values (scalar or (ndarray, Tensor)): values to be padded if not pad_same

    Returns:
        padded sequence (np.ndarray or torch.Tensor)
    """
    assert isinstance(seq, (np.ndarray, torch.Tensor))
    assert pad_same or pad_values is not None
    if pad_values is not None:
        assert isinstance(pad_values, float)
    repeat_func = np.repeat if isinstance(seq, np.ndarray) else torch.repeat_interleave
    concat_func = np.concatenate if isinstance(seq, np.ndarray) else torch.cat
    ones_like_func = np.ones_like if isinstance(seq, np.ndarray) else torch.ones_like
    seq_dim = 1 if batched else 0

    begin_pad = []
    end_pad = []

    if padding[0] > 0:
        pad = seq[[0]] if pad_same else ones_like_func(seq[[0]]) * pad_values
        begin_pad.append(repeat_func(pad, padding[0], seq_dim))
    if padding[1] > 0:
        pad = seq[[-1]] if pad_same else ones_like_func(seq[[-1]]) * pad_values
        end_pad.append(repeat_func(pad, padding[1], seq_dim))

    return concat_func(begin_pad + [seq] + end_pad, seq_dim)        


def pad_sequence(seq, padding, batched=False, pad_same=True, pad_values=None):
    """
    Pad a nested dictionary or list or tuple of sequence tensors in the time dimension (dimension 1).

    Args:
        seq (dict or list or tuple): a possibly nested dictionary or list or tuple with tensors
            of leading dimensions [B, T, ...]
        padding (tuple): begin and end padding, e.g. [1, 1] pads both begin and end of the sequence by 1
        batched (bool): if sequence has the batch dimension
        pad_same (bool): if pad by duplicating
        pad_values (scalar or (ndarray, Tensor)): values to be padded if not pad_same

    Returns:
        padded sequence (dict or list or tuple)
    """
    return recursive_dict_list_tuple_apply(
        seq,
        {
            torch.Tensor: lambda x, p=padding, b=batched, ps=pad_same, pv=pad_values:
                pad_sequence_single(x, p, b, ps, pv),
            np.ndarray: lambda x, p=padding, b=batched, ps=pad_same, pv=pad_values:
                pad_sequence_single(x, p, b, ps, pv),
            type(None): lambda x: x,
        }
    )

def partition_dataset_train_valid(dataset):
    np.random.seed(0)
    random.seed(0)
    hdf5_file_path = dataset
    file = h5py.File(hdf5_file_path, 'a')
    all_keys = list(file['data'].keys())

    # Shuffle the keys
    random.shuffle(all_keys)
    print("all_keys: ", all_keys)

    # Calculate the number of test samples (10% of the total data)
    num_test_samples = int(len(all_keys) * 0.1)

    # Split the keys into train and test sets
    test_keys = all_keys[:num_test_samples]
    test_keys = [np.bytes_(s) for s in test_keys]
    train_keys = all_keys[num_test_samples:]
    train_keys = [np.bytes_(s) for s in train_keys]

    # Print the results
    print(f"Train keys: {len(train_keys)}")
    print(f"Test keys: {len(test_keys)}")
    
    file = h5py.File(hdf5_file_path, 'a')
    group_key = 'mask'
    if group_key not in file:
        group = file.create_group(group_key)
    group = file[group_key]

    group.create_dataset('train', data=train_keys, compression='gzip', compression_opts=9)
    group.create_dataset('valid', data=test_keys, compression='gzip', compression_opts=9)


# Utilitiy functions for copying / editing HDF5 files
        
def copy_attrs(src, dst):
    """ Copy attributes from one HDF5 object to another """
    for key, value in src.attrs.items():
        dst.attrs[key] = value

def copy_group(source_group, dest_group):
    for key in source_group:
        item = source_group[key]
        print("key, type(item): ", key, type(item))
        if isinstance(item, h5py.Group):
            new_group = dest_group.create_group(key)
            copy_attrs(item, new_group)
            copy_group(item, new_group)

        elif isinstance(item, h5py.Dataset):
            dest_group.create_dataset(key, data=item[()], compression='gzip', compression_opts=9)
            copy_attrs(item, dest_group)

def edit_and_merge_hdf5(old_hdf5, new_hdf5):
    new_file = h5py.File(new_hdf5, 'a')
    group_key = 'data'
    if group_key not in new_file:
        group = new_file.create_group(group_key)
    group = new_file[group_key]

    old_file = h5py.File(old_hdf5, 'r')
    copy_group(old_file, group)    


