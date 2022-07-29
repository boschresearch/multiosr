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


def get_all_attrs(dataset, ds_attribute_index):
    attrs_set = set()
    for i in range(len(dataset)):
        data = dataset[i]
        attrs_set.add(tuple(data[ds_attribute_index].numpy()))

    return sorted(list(attrs_set))


def get_all_attr_cls_ids_list(dataset, ds_attribute_index):
    num_attributes = len(dataset[0][ds_attribute_index])
    attr_cls_ids_set_list = [set() for _ in range(num_attributes)]
    for i in range(len(dataset)):
        data = dataset[i]

        for j in range(num_attributes):
            attr_cls_id = int(data[ds_attribute_index][j])
            attr_cls_ids_set_list[j].add(attr_cls_id)

    attr_cls_ids_list = [sorted(list(attr_cls_ids_set)) for attr_cls_ids_set in attr_cls_ids_set_list]
    return attr_cls_ids_list
