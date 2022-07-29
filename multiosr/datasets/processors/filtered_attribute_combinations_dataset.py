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


from copy import deepcopy

from .base_dataset import BaseDataset


class FilterAttributeCombinationsDataset(BaseDataset):
    def __init__(self, parent_dataset, valid_combinations, ds_attribute_index):
        super().__init__(parent_dataset)

        self.valid_combinations = set(valid_combinations)

        self.valid_indices = []
        for index in range(len(parent_dataset)):
            sample_combination = tuple(self.parent_dataset[index][ds_attribute_index].cpu().numpy())
            if sample_combination in self.valid_combinations:
                self.valid_indices.append(index)

    def __getitem__(self, i):
        return self.parent_dataset[self.valid_indices[i]]

    def __len__(self):
        return len(self.valid_indices)

    def get_old_to_new_attr_cls_dict_list(self):
        return deepcopy(self.old_to_new_attr_cls_dict_list)
