# The code accompanying the the paper "Multi-Attribute Open Set Recognition" accepted at GCPR 2022.
# Copyright (c) 2022 Robert Bosch GmbH
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Piyapat Saranrittichai, Volker Fischer
# -*- coding: utf-8 -*-

import json
import math
import pickle
import torch


def load_json(json_file):
    with open(json_file) as f:
        all_config_data = json.load(f)
    return all_config_data


def store_json(data_dict, json_file):
    with open(json_file, 'w') as f:
        json.dump(data_dict, f, indent=4, sort_keys=True)


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


class MeanAccumulator(object):
    def __init__(self):
        self.current_mean = 0
        self.current_count = 0

    def accumulate(self, val, count):
        new_count = self.current_count + count
        self.current_mean = self.current_mean * (self.current_count / new_count) + val * (count / new_count)
        self.current_count = new_count

    def get_mean(self):
        return self.current_mean


class MeanAccumulatorSet(object):
    def __init__(self, var_names=None):
        self.name_accumulator_dict = None
        if var_names is not None:
            self.reset_name_accumulator_dict(var_names)

    def accumulate(self, name_val_dict, count):
        for name, val in name_val_dict.items():
            self.name_accumulator_dict[name].accumulate(val, count)

    def get_name_mean_dict(self):
        name_mean_dict = {}
        for name in self.name_accumulator_dict.keys():
            name_mean_dict[name] = self.name_accumulator_dict[name].get_mean()

        return name_mean_dict

    def reset_name_accumulator_dict(self, var_names):
        self.name_accumulator_dict = {name: MeanAccumulator() for name in var_names}


class MaxTrackerSet(object):
    def __init__(self, var_names):
        self.name_max_dict = {name: -math.inf for name in var_names}

    def update(self, name_val_dict):
        for name, val in name_val_dict.items():
            if val > self.name_max_dict[name]:
                self.name_max_dict[name] = val

    def get_name_max_dict(self):
        return self.name_max_dict


def get_iid_sample_indices(attrs_batch, attr_num_ids_list):
    iid_sample_indices = []
    for i in range(attrs_batch.shape[0]):
        is_iid = all([(y < y_max).item() for y, y_max in zip(attrs_batch[i], attr_num_ids_list)])
        if is_iid:
            iid_sample_indices.append(i)

    return iid_sample_indices


def batch_wise_to_sample_wise_data(input_list):
    if type(input_list[0]) is dict:
        output_data = {}
        for k in input_list[0].keys():
            output_data[k] = torch.cat(
                [input_list[i][k] for i in range(len(input_list))])
    else:
        output_data = []
        for k in range(len(input_list[0])):
            output_data.append(
                torch.cat([input_list[i][k] for i in range(len(input_list))])
            )

    return output_data
