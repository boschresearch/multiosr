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

import torch
import numpy as np
import os
import random
import pickle


def save_pkl(data, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


def save_list_to_file(input_list, output_path):
    with open(output_path, 'w') as f:
        for data in input_list:
            f.write('{}\n'.format(data))


def extract_metadata(data):
    img_path_list = []
    img_label_names_list = []
    pair_img_list_dict = {}
    for index, instance in enumerate(data):
        image, attr, obj = instance['image'], instance['attr'], instance['obj']

        if (attr, obj) not in pair_img_list_dict:
            pair_img_list_dict[(attr, obj)] = []
        pair_img_list_dict[(attr, obj)].append(image)
        img_path_list.append(image)
        img_label_names_list.append((attr, obj))

    all_material_names = sorted(list(set([material for material, _ in pair_img_list_dict.keys()])))
    all_type_names = sorted(list(set([type for _, type in pair_img_list_dict.keys()])))
    all_material_to_id_dict = {n: i for i, n in enumerate(all_material_names)}
    all_type_to_id_dict = {n: i for i, n in enumerate(all_type_names)}

    img_labels = \
        [(all_material_to_id_dict[label_names[0]], all_type_to_id_dict[label_names[1]]) for label_names in img_label_names_list]

    return pair_img_list_dict, img_path_list, np.array(img_labels), all_material_names, all_type_names


if __name__ == '__main__':
    random.seed(0)

    # load metadata
    dataset_dir = "ut-zappos-material"
    metadata = torch.load(os.path.join(dataset_dir, 'metadata.t7'))
    pair_img_list_dict, all_img_path_list, all_img_labels, all_material_names, all_type_names = extract_metadata(metadata)

    selected_type_names = ['Boots.Knee.High', 'Boots.Mid-Calf', 'Shoes.Flats', 'Shoes.Heels', 'Shoes.Loafers',
                           'Boots.Ankle', 'Sandals', 'Shoes.Oxfords', 'Shoes.Sneakers.and.Athletic.Shoes']
    selected_material_names = ["Faux.Leather", "Full.grain.leather", "Leather", "Rubber", "Suede", "Canvas", "Nubuck",
                               "Patent.Leather", "Satin", "Synthetic"]

    selected_train_combinations = {
        'osr': [("Faux.Leather", 'Boots.Knee.High'), ("Faux.Leather", 'Boots.Mid-Calf'), ("Faux.Leather", 'Shoes.Flats'),
                ("Full.grain.leather", 'Boots.Mid-Calf'), ("Full.grain.leather", 'Shoes.Loafers'),
                ("Leather", 'Shoes.Flats'), ("Leather", 'Shoes.Heels'), ("Leather", 'Shoes.Loafers'),
                ("Rubber", 'Boots.Knee.High'), ("Rubber", 'Boots.Mid-Calf'),
                ("Suede", 'Boots.Knee.High'), ("Suede", 'Shoes.Flats'), ("Suede", 'Shoes.Heels')]
    }

    selected_type_name_to_index_dict = {n: i for i, n in enumerate(selected_type_names)}
    selected_material_name_to_index_dict = {n: i for i, n in enumerate(selected_material_names)}

    # split
    max_pair_n_samples = 300

    for split_name, train_combinations in selected_train_combinations.items():
        img_train_list, img_val_list, img_test_list = [], [], []

        for pair, img_list in pair_img_list_dict.items():
            if (pair[0] not in selected_material_names) or (pair[1] not in selected_type_names):
                continue

            # crop max pair samples
            img_list_shuffle = random.sample(img_list, len(img_list))[:max_pair_n_samples]

            if pair in train_combinations:
                n_train = int(len(img_list_shuffle) * 0.8)
                n_val = int(len(img_list_shuffle) * 0.1)
                n_test = len(img_list_shuffle) - n_train - n_val

                img_train_list.extend(img_list_shuffle[:n_train])
                img_val_list.extend(img_list_shuffle[n_train:(n_train + n_val)])
                img_test_list.extend(img_list_shuffle[(n_train + n_val):(n_train + n_val + n_test)])
            else:
                n_test = int(len(img_list_shuffle) * 0.1)
                img_test_list.extend(img_list_shuffle[:n_test])

        # save split
        split_dir = f'split_{split_name}'
        os.makedirs(os.path.join(dataset_dir, split_dir), exist_ok=True)
        save_list_to_file(img_train_list, os.path.join(dataset_dir, split_dir, 'train.txt'))
        save_list_to_file(img_val_list, os.path.join(dataset_dir, split_dir, 'val.txt'))
        save_list_to_file(img_test_list, os.path.join(dataset_dir, split_dir, 'test.txt'))
        save_list_to_file(selected_material_names,
                          os.path.join(dataset_dir, split_dir, '0_selected_material_names.txt'))
        save_list_to_file(selected_type_names, os.path.join(dataset_dir, split_dir, '1_selected_type_names.txt'))
        print('N Train: {} / N Val: {} / N Test: {}'.format(len(img_train_list), len(img_val_list), len(img_test_list)))

        # save literals
        attr_index_literal_labels_dict = {
            0: all_material_names,
            1: all_type_names
        }
        save_pkl({'attr_index_literal_labels_dict': attr_index_literal_labels_dict},
                 os.path.join(dataset_dir, 'literals.pkl'))
