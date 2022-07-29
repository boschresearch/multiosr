import os
import numpy as np
import h5py

import torch
from torch.utils.data import Dataset


def get_group_data_dict(root, phase):
    if phase == 'train':
        group_filename_dict = {
            'train': os.path.join(root, 'train.h5py')
        }
    elif phase == 'val':
        group_filename_dict = {
            'val': os.path.join(root, 'val.h5py')
        }
    else:
        group_filename_dict = {
            'test': os.path.join(root, 'test.h5py')
        }

    group_data_dict = {group: h5py.File(filename) for group, filename in group_filename_dict.items()}
    return group_data_dict


class ObjectBackgroundCombinationDataset(Dataset):
    def __init__(self, root, phase):
        self.phase = phase
        self.group_data_dict = get_group_data_dict(root, phase)

        all_keys = sorted(list(self.group_data_dict.keys()))
        all_sizes = [self.group_data_dict[key]['attributes'].shape[0] for key in all_keys]

        self.data_key_list, self.data_index_list = [], []
        for key, size in zip(all_keys, all_sizes):
            self.data_key_list += [key] * size
            self.data_index_list += np.arange(size).tolist()

    def __getitem__(self, i):
        key, index = self.data_key_list[i], self.data_index_list[i]

        image = self.group_data_dict[key]['images'][index:index+1, ...][0]
        y = self.group_data_dict[key]['attributes'][index:index + 1, ...][0]

        return torch.tensor(image, dtype=torch.float32), torch.tensor([*y]).long()

    def __len__(self):
        return len(self.data_key_list)

    def get_config(self):
        config = {
            'ds_x_index': 0,
            'ds_attribute_index': 1
        }
        return config
