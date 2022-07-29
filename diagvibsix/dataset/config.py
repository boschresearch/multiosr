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
# Author: Elias Eulig, Piyapat Saranrittichai, Volker Fischer
# -*- coding: utf-8 -*-

import os
import numpy as np

__all__ = ['TEXTURES', 'DATASETS', 'CATEGORIES', 'SATURATION', 'COLORS', 'COL_LIGHTNESS', 'TEXT_LIGHTNESS', 'POSITION', 'VELOCITY',
           'ORIENTATION', 'ROTATION', 'SCALE', 'SCALING', 'OBJECT_ATTRIBUTES']

DATASET_DIR = '/home/asp2abt/workspace/multiosr/data/diagvibsix'


def load_dataset(DATASETS):
    """
    For a given dataset dict load the mnist, fashion-mnist and shapes data.

    Parameters
    ----------
    DATASET : dict
        DATASET dictionary that contains for each dataset the dataset savepath. Must be a numpy .npz file.

    Returns
    -------
    dict
        Dataset dictionary that contains an additional key 'X' for each dataset with the dataset.
    """

    for d_name, d in DATASETS.items():
        if 'X' in d:
            # skip already loaded dataset
            continue
        if 'mnist' in d_name:
            name, split = d_name.split()
            x = np.load(d['savepath'])['x_' + split]
            y = np.load(d['savepath'])['y_' + split]
            sorted_idxs = y.argsort()
            x = x[sorted_idxs]
        else:
            raise ValueError('Dataset {} unknown'.format(d_name))
        d['X'] = x
    return DATASETS


# TEXTURES
textures_names = ['tiles', 'wood', 'carpet', 'bricks', 'lava']

textures_savepath = os.path.join(DATASET_DIR, 'textures')
TEXTURES = {name: os.path.join(textures_savepath, name + '.png') for name in textures_names}

# DATASETS
DATASETS = {
    'mnist train': {
        'classes': [str(i) for i in range(10)],
        'samples': [4738, 5393, 4766, 4904, 4673, 4336, 4734, 5012, 4680, 4759],
        'savepath': os.path.join(DATASET_DIR, 'processed', 'mnist.npz'),
        'size': 40
    },
    'mnist val': {
        'classes': [str(i) for i in range(10)],
        'samples': [1185, 1349, 1192, 1227, 1169, 1085, 1184, 1253, 1171, 1190],
        'savepath': os.path.join(DATASET_DIR, 'processed', 'mnist.npz'),
        'size': 40
    },
    'mnist test': {
        'classes': [str(i) for i in range(10)],
        'samples': [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009],
        'savepath': os.path.join(DATASET_DIR, 'processed', 'mnist.npz'),
        'size': 40
    }
}

# QUESTIONS AND ATTRIBUTES
## Categories
CATEGORIES = ['mnist']

## Colors
# We define colors in HSL colorspace
SATURATION = 1.
COL_LIGHTNESS = {'dark': (1/7, 2/7), 'penumbra': (3/7, 4/7), 'bright': (5/7, 6/7)}
TEXT_LIGHTNESS = {'dark': ((0., 1/11), (4/11, 5/11)), 'darker': ((2/11, 3/11), (6/11, 7/11)),
                  'brighter': ((4/11, 5/11), (8/11, 9/11)), 'bright': ((6/11, 7/11), (10/11, 1.))}

hue_values = {
    'red': 0, 'orange': 30, 'yellow': 60, 
    'chartreuse_green': 90, 'green': 120, 'spring_green': 150, 
    'cyan': 180, 'azure': 210, 'blue': 240, 
    'violet': 270, 'magenta': 300, 'rose': 330}
COLORS = {name: (v - 15, v + 15) for name, v in hue_values.items()}

## Positions & VELOCITY
pos_left_right = {'left': (1/7, 2/7), 'center': (3/7, 4/7), 'right': (5/7, 6/7)}
pos_up_down = {'upper': (1/7, 2/7), 'center': (3/7, 4/7), 'lower': (5/7, 6/7)}
POSITION = {'{} {}'.format(y_sem, x_sem): (dy, dx) for y_sem, dy in pos_up_down.items() for x_sem, dx in pos_left_right.items()}

v_left_right = {'left': (-0.1, 0), 'none': (0., 0.), 'right': (0., 0.1)}
v_up_down = {'up': (-0.1, 0), 'none': (0., 0.), 'down': (0., 0.1)}
VELOCITY = {'{} {}'.format(y_sem, x_sem): (dy, dx) for y_sem, dy in v_up_down.items() for x_sem, dx in v_left_right.items()}

## Orientation & Rotation
ORIENTATION = {'upright': (0, 0),
               'left': ((3 / 8) * np.pi, (5 / 8) * np.pi),
               'right': ((11 / 8) * np.pi, (13 / 8) * np.pi),
               'upside down': ((7 / 8) * np.pi, (9 / 8) * np.pi)}

ROTATION = {'clockwise': (-np.pi/2, 0.),
            'no rotation': (0, 0),
            'counterclockwise': (0, np.pi/2)}

## Scale & Scaling
SCALE = {'small': (1/1.45, 1/1.35), 'smaller': (1/1.25, 1/1.15), 'normal': (1/1.05, 1.05),
         'larger': (1.15, 1.25), 'large': (1.35, 1.45)}
SCALING = {'smaller': (-0.1, -0.05), 'no scaling': (0., 0.), 'larger': (0.05, 0.1)}

OBJECT_ATTRIBUTES = {
    'category': CATEGORIES,
    'shape': list(range(max([len(DATASETS[d]['classes']) for d in DATASETS.keys()]))),
    'style':
        [
            'color',
            'texture',
            'full_texture',
            'boundary',
            'noisy static',
            'noisy dynamic',
        ],
    'hue': list(COLORS),
    'lightness': list(COL_LIGHTNESS),
    'texture': textures_names,
    'orientation': list(ORIENTATION),
    'rotation': list(ROTATION),
    'position': list(POSITION),
    'velocity': list(VELOCITY),
    'scale': list(SCALE),
    'scaling': list(SCALING)
}

ATTRIBUTE_VECTOR = [attr for obj_attributes in OBJECT_ATTRIBUTES.values() for attr in obj_attributes]
