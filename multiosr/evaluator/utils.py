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


import numpy as np


def compute_auc(X, Y):
    auc = 0
    x_prev, y_prev = 0, 0
    for x, y in zip(X, Y):
        dx = x - x_prev
        dy = y - y_prev

        area = dx * (y_prev + 0.5 * dy)
        auc += area

        x_prev, y_prev = x, y
    return auc


# normal: y_true[i] = 1, y_score = high
# abnormal: y_true[i] = 0, y_score = low
def compute_label_wise_positive_rate_dict(y_true, y_score):
    # sort descendingly
    sort_indices = np.argsort(-y_score) # negative sign to sort ascendingly
    y_true_sort = y_true[sort_indices].astype(int)
    y_score_sort = y_score[sort_indices]

    # compute Positive Rate
    id_num_samples_dict = dict(zip(*np.unique(y_true_sort, return_counts=True)))
    id_counter_dict = {id: 0 for id in id_num_samples_dict.keys()}
    id_prlist_dict = {id: [0] for id in id_num_samples_dict.keys()}

    for i in range(len(y_true_sort)):
        sample_id = y_true_sort[i]
        id_counter_dict[sample_id] += 1

        for id in id_prlist_dict.keys():
            pr = id_counter_dict[id] / id_num_samples_dict[id]
            id_prlist_dict[id].append(pr)

    return id_prlist_dict
