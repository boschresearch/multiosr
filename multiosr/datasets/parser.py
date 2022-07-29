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
import logging

from . import wrappers as wrappers
from . import processors as processors
from . import utils as utils


def get_dataset_catalog(dataset_spec):
    dataset_catalog = {}

    # targets
    existing_attr_cls_ids_list_raw = None
    for target_key in ['target_train', 'target_val', 'target_test']:
        dataset_catalog[target_key] = get_dataset(dataset_spec[target_key], existing_attr_cls_ids_list_raw)

        if 'attr_cls_ids_list_raw' in dataset_catalog[target_key][1]:
            existing_attr_cls_ids_list_raw = dataset_catalog[target_key][1]['attr_cls_ids_list_raw_ordered']

        # get available combinations
        dataset, dataset_config = dataset_catalog[target_key]
        available_combinations = utils.get_all_attrs(dataset, dataset_config['ds_attribute_index'])
        dataset_catalog[target_key][1]['available_combinations'] = available_combinations

        # log
        num_samples = len(dataset_catalog[target_key][0])
        logging.info(f'Dataset[{target_key}] has {num_samples} samples')

    return dataset_catalog


def get_dataset(dataset_params, existing_attr_cls_ids_list_raw):
    # get dataset
    if dataset_params['name'] == 'diag_vib':
        dataset_raw = wrappers.DiagVibDataset(**dataset_params['params'])
        dataset_config = dataset_raw.get_config()
    elif dataset_params['name'] == 'object_background_combination':
        dataset_raw = wrappers.ObjectBackgroundCombinationDataset(**dataset_params['params'])
        dataset_config = dataset_raw.get_config()
    elif dataset_params['name'] == 'ut_zappos':
        dataset_raw = wrappers.UTZapposDataset(**dataset_params['params'])
        dataset_config = dataset_raw.get_config()
    else:
        raise RuntimeError('Dataset with name {} is not supported.'.format(dataset_params['name']))

    # get proper attribute order
    attr_cls_ids_list_raw = utils.get_all_attr_cls_ids_list(dataset_raw, dataset_config['ds_attribute_index'])
    if existing_attr_cls_ids_list_raw is not None:
        attr_cls_ids_list_raw_ordered = deepcopy(existing_attr_cls_ids_list_raw)

        num_attributes = len(existing_attr_cls_ids_list_raw)
        for i in range(num_attributes):
            new_attr_cls_ids = sorted(list(set(attr_cls_ids_list_raw[i]).difference(existing_attr_cls_ids_list_raw[i])))
            attr_cls_ids_list_raw_ordered[i].extend(new_attr_cls_ids)
    else:
        attr_cls_ids_list_raw_ordered = attr_cls_ids_list_raw

    # order attribute ids
    dataset = processors.OrderedAttributesDataset(dataset_raw, attr_cls_ids_list_raw_ordered,
                                                  dataset_config['ds_attribute_index'])

    # update config
    dataset_config['attr_cls_ids_list_raw'] = attr_cls_ids_list_raw
    dataset_config['attr_cls_ids_list_raw_ordered'] = attr_cls_ids_list_raw_ordered

    return dataset, dataset_config
