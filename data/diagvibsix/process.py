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

import numpy as np
from skimage.transform import rescale
from tqdm import trange


def get_bbox(im, idx=255):
    """Returns bounding box.
    """

    ys = np.where(np.max(im, axis=0) == idx)
    xs = np.where(np.max(im, axis=1) == idx)
    bbox = (slice(np.min(xs), np.max(xs) + 1), slice(np.min(ys), np.max(ys) + 1))
    return bbox


def process_mnist():
    SHARED_LOADPATH_MNIST = "raw/mnist.npz"
    SHARED_SAVEPATH_MNIST = "processed/mnist.npz"

    IMG_SIZE = 40
    OBJ_SIZE = IMG_SIZE // np.sqrt(2)
    TRAIN_VAL_SPLIT = 0.8
    data = np.load(SHARED_LOADPATH_MNIST)

    processed_data = {'x_train': [None] * 60000, 'x_test': [None] * 10000,
                      'y_train': data['y_train'], 'y_test': data['y_test']}

    for file in data.files:
        if file.startswith('x'):
            dataset = data[file]
            print('Preprocess {} ...'.format(file))
            for idx in trange(dataset.shape[0]):
                im = dataset[idx]
                mask = np.where(im > 100, 255, 0)
                bbox = get_bbox(mask, idx=255)
                im = im[bbox]
                im_rescaled = rescale(im, np.min([OBJ_SIZE / im.shape[0], OBJ_SIZE / im.shape[1]]), anti_aliasing=True,
                                      preserve_range=True, order=3).astype('uint8')
                canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype='uint8')
                x0 = int((IMG_SIZE - 1) / 2 - (im_rescaled.shape[1] - 1) / 2)
                y0 = int((IMG_SIZE - 1) / 2 - (im_rescaled.shape[0] - 1) / 2)
                pos = (slice(y0, y0 + im_rescaled.shape[0]), slice(x0, x0 + im_rescaled.shape[1]))
                canvas[pos] = im_rescaled
                processed_data[file][idx] = canvas

    processed_data['x_train'] = np.array(processed_data['x_train'])
    processed_data['x_test'] = np.array(processed_data['x_test'])

    """ Split training data in 80% train and 20% validation """
    x_train = []
    x_val = []
    samples_train = []
    samples_val = []

    sorted_train_idxs = processed_data['y_train'].argsort()
    x_train_sorted = processed_data['x_train'][sorted_train_idxs]
    y_train_sorted = processed_data['y_train'][sorted_train_idxs]
    train_samples = [np.sum(y_train_sorted == i) for i in range(10)]

    sorted_test_idxs = processed_data['y_test'].argsort()
    x_test = processed_data['x_test'][sorted_test_idxs]
    y_test = processed_data['y_test'][sorted_test_idxs]

    for i in range(10):
        begin_train = int(np.sum(train_samples[:i]))
        split = int(train_samples[i] * TRAIN_VAL_SPLIT + np.sum(train_samples[:i]))
        end_val = int(np.sum(train_samples[:i + 1]))

        x_train.append(x_train_sorted[slice(begin_train, split)])
        x_val.append(x_train_sorted[slice(split, end_val)])
        samples_train.append(x_train[i].shape[0])
        samples_val.append(x_val[i].shape[0])

    x_train = np.concatenate(x_train)
    x_val = np.concatenate(x_val)
    y_train = np.array([i for i in range(10) for j in range(samples_train[i])], dtype='uint8')
    y_val = np.array([i for i in range(10) for j in range(samples_val[i])], dtype='uint8')

    np.savez(SHARED_SAVEPATH_MNIST,
             x_train=x_train,
             x_val=x_val,
             x_test=x_test,
             y_train=y_train,
             y_val=y_val,
             y_test=y_test)


if __name__ == '__main__':
    process_mnist()
