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
from tqdm import tqdm
from . import utils


# normal: y_true[i] = 1, y_score = high
# abnormal: y_true[i] = 0, y_score = low
def compute_auroc(y_true, y_score):
    # compute positive rate
    id_prlist_dict = utils.compute_label_wise_positive_rate_dict(y_true, y_score)

    # compute AUROC
    Y = np.array(id_prlist_dict[1]) # TPR
    X = np.array(id_prlist_dict[0]) if 0 in id_prlist_dict else np.zeros_like(Y) # FPR

    X = np.array([0.0] + np.sort(X).tolist() + [1.0])
    Y = np.array([0.0] + np.sort(Y).tolist() + [1.0])
    auroc = utils.compute_auc(X, Y)

    TPR_arr = Y
    FPR_arr = X
    return auroc, TPR_arr, FPR_arr


# y_score (high=>normal, low=>novel)
def compute_open_set_optimal_threshold(y_pred, y_score, y_gt, uuc_label_id=None):
    if uuc_label_id is None:
        uuc_label_id = max(y_gt)

    # get possible thresolds
    sorted_score = sorted(set(y_score))
    possible_thresholds = [sorted_score[0] - 1e-2]
    for i in range(len(sorted_score)-1):
        possible_thresholds.append((sorted_score[i] + sorted_score[i+1])/2.0)
    possible_thresholds.append(sorted_score[-1] + 1e-2)

    # compute best threshold
    best_threshold, best_f_measure = 0, -1
    for score_threshold in tqdm(possible_thresholds, desc="Computing Open-Set Threshold"):
        f_measure, _, _ = compute_open_set_f_measure(y_pred, y_score, y_gt, score_threshold, uuc_label_id)

        if best_f_measure < f_measure:
            best_f_measure = f_measure
            best_threshold = score_threshold

    return best_threshold, best_f_measure


def compute_open_set_f_measure(y_pred, y_score, y_gt, score_threshold, uuc_label_id=None):
    if uuc_label_id is None:
        uuc_label_id = max(y_gt)

    # treat low score prediction as UUC prediction
    y_pred_proc = np.copy(y_pred)
    for i in range(y_pred_proc.shape[0]):
        if y_score[i] < score_threshold:
            y_pred_proc[i] = uuc_label_id

    # compute TP, FP, FN
    label_id_tp_dict, label_id_fp_dict, label_id_fn_dict = \
        compute_tp_fp_fn_all_classes(y_pred_proc, y_gt)

    # compute precision/recall
    num_kkc_classes = len(set(y_gt)) - 1
    precision = compute_precision(label_id_tp_dict, label_id_fp_dict, num_kkc_classes, 'micro')
    recall = compute_recall(label_id_tp_dict, label_id_fn_dict, num_kkc_classes, 'micro')

    # compute f_measure
    if precision + recall == 0:
        f_measure = 0
    else:
        f_measure = (2 * precision * recall) / (precision + recall)

    return f_measure, precision, recall


def compute_tp_fp_fn_all_classes(y_pred, y_gt):
    label_id_tp_dict = {gt_id: 0 for gt_id in set(y_gt)}
    label_id_fp_dict = {gt_id: 0 for gt_id in set(y_gt)}
    label_id_fn_dict = {gt_id: 0 for gt_id in set(y_gt)}
    for pred_id, gt_id in zip(y_pred, y_gt):
        if pred_id == gt_id:
            label_id_tp_dict[gt_id] += 1
        else:
            label_id_fp_dict[pred_id] += 1
            label_id_fn_dict[gt_id] += 1

    return label_id_tp_dict, label_id_fp_dict, label_id_fn_dict


def compute_precision(label_id_tp_dict, label_id_fp_dict, num_kkc_classes, type):
    return compute_pc_helper(label_id_tp_dict, label_id_fp_dict, num_kkc_classes, type)


def compute_recall(label_id_tp_dict, label_id_fn_dict, num_kkc_classes, type):
    return compute_pc_helper(label_id_tp_dict, label_id_fn_dict, num_kkc_classes, type)


def compute_pc_helper(label_id_tp_dict, label_id_fx_dict, num_kkc_classes, type):
    assert type in ['macro', 'micro']
    if type == 'macro':
        val = 0
        for label_id in range(num_kkc_classes):
            val += label_id_tp_dict[label_id]/(label_id_tp_dict[label_id] + label_id_fx_dict[label_id])
        return val / num_kkc_classes
    elif type == 'micro':
        val_num, val_denom = 0, 0
        for label_id in label_id_tp_dict.keys():
            val_num += label_id_tp_dict[label_id]
            val_denom += (label_id_tp_dict[label_id] + label_id_fx_dict[label_id])

        if val_num == 0:
            return 0
        else:
            return val_num / val_denom


def refine_y_pred_with_score_threshold(y_pred, y_score, y_gt, score_threshold, uuc_label_id=None):
    if uuc_label_id is None:
        uuc_label_id = max(y_gt)

    # treat low score prediction as UUC prediction
    y_pred_proc = np.copy(y_pred)
    for i in range(y_pred_proc.shape[0]):
        if y_score[i] < score_threshold:
            y_pred_proc[i] = uuc_label_id

    return y_pred_proc


# The following function 'compute_oscr' is from ARPL
#   (https://github.com/iCGY96/ARPL/blob/master/core/evaluation.py)
# Copyright (c) 2020 Guangyao Chen, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def compute_oscr(pred_k, pred_u, labels):
    x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    pred = np.argmax(pred_k, axis=1)
    correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for _ in range(n + 2)]
    FPR = [0 for _ in range(n + 2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2)) if len(x2) > 0 else 0

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w

    return OSCR
