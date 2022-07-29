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


import os
import numpy as np

from . import evaluator
from ..utils import plotting


class Metrics(object):
    def __init__(self, phase_name, train_attr_num_ids_list, train_attr_combinations, plot_dir):
        self.phase_name = phase_name
        self.train_attr_num_ids_list = train_attr_num_ids_list
        self.train_attr_combinations = set(train_attr_combinations)
        self.plot_dir = plot_dir

        self.num_attrs = len(self.train_attr_num_ids_list)

    def process(self, all_pred_raw_output, all_pred_conf_raw_output, all_attr_gt, compute_threshold=False):
        assert (len(all_pred_raw_output) == self.num_attrs)
        assert (len(all_pred_conf_raw_output) == self.num_attrs)
        assert (all_attr_gt.shape[1] == self.num_attrs)

        metrics_dict = {}

        all_pred_output = []
        all_pred_conf_output = []
        attr_gt_clipped_list = []
        attr_score_threshold_list = []

        # compute per-attribute metrics
        for i in range(self.num_attrs):
            # assign gt of unknown classes to have the same labels
            attr_num_ids = self.train_attr_num_ids_list[i]
            attr_gt_clipped = np.clip(all_attr_gt[:, i].cpu().numpy(), 0, attr_num_ids)

            # get prediction
            attr_pred = all_pred_raw_output[i].cpu().numpy().argmax(axis=1)
            attr_pred_conf = all_pred_conf_raw_output[i].cpu().numpy()

            # append to lists
            all_pred_output.append(attr_pred)
            all_pred_conf_output.append(attr_pred_conf)
            attr_gt_clipped_list.append(attr_gt_clipped)

            # compute metrics: close-set accuracy
            close_set_indices = [index for index, label in enumerate(attr_gt_clipped) if label < attr_num_ids]
            close_set_accuracy = np.equal(attr_pred[close_set_indices], attr_gt_clipped[close_set_indices]).mean()
            metrics_dict[f'close_set_accuracy_{i}'] = close_set_accuracy

            # compute metrics: AUROC
            test_y_true_ad = np.array([0 if y == attr_num_ids else 1 for y in attr_gt_clipped])
            test_auroc, TPR_arr, FPR_arr = evaluator.compute_auroc(test_y_true_ad, attr_pred_conf)
            metrics_dict[f'open_set_auroc_{i}'] = test_auroc

            savepath = os.path.join(self.plot_dir,
                                    f'Evaluation_Phase_{self.phase_name}_AUROC_Curve_{i}.jpg')
            plotting.plot_XY_curve(FPR_arr, TPR_arr, 'FPR', 'TPR', savepath)

            # compute metrics: OSCR
            open_set_indices = [index for index, label in enumerate(attr_gt_clipped) if label >= attr_num_ids]
            pred_k = all_pred_raw_output[i].cpu().numpy()[close_set_indices, :]
            pred_u = all_pred_raw_output[i].cpu().numpy()[open_set_indices, :]
            labels = attr_gt_clipped[close_set_indices]
            oscr = evaluator.compute_oscr(pred_k, pred_u, labels)
            metrics_dict[f'oscr_{i}'] = oscr

            if compute_threshold:
                # compute thresholds with respect to F-Measure score
                open_set_score_threshold, _ = evaluator.compute_open_set_optimal_threshold(
                    attr_pred, attr_pred_conf, attr_gt_clipped, uuc_label_id=attr_num_ids)

                attr_score_threshold_list.append(open_set_score_threshold)

        # merge metrics across attributes
        for m in ['close_set_accuracy', 'open_set_auroc', 'oscr']:
            if f'{m}_0' in metrics_dict:
                val_list = [metrics_dict[f'{m}_{i}'] for i in range(self.num_attrs)]
                metrics_dict[f'{m}_avg'] = np.mean(val_list)

        # plotting:  cross-factor anomaly influence
        self.plot_cross_attribute_confidence_scores(all_pred_conf_output, all_attr_gt)

        # compute OE
        if len(attr_score_threshold_list) == self.num_attrs:
            # refine prediction with thresholds
            all_pred_output_extended = self.refine_prediction_with_thresholds(all_pred_output, all_pred_conf_output,
                                                                              attr_score_threshold_list)
            # compute OE metric
            self.plot_oe_heatmap(all_pred_output_extended, all_attr_gt)

        return metrics_dict

    def plot_oe_heatmap(self, all_pred_output, all_attr_gt):
        all_pred_output = np.array(all_pred_output).transpose((1, 0))

        num_attrs = len(self.train_attr_num_ids_list)
        num_entries = np.power(2, num_attrs)
        cmat = np.zeros((num_entries, num_entries)).astype(np.int)
        for i in range(all_attr_gt.shape[0]):
            ood_gt = [0 if id.item() < num_ids else 1 for id, num_ids in
                      zip(all_attr_gt[i], self.train_attr_num_ids_list)]
            ood_pred = [0 if id.item() < num_ids else 1 for id, num_ids in
                        zip(all_pred_output[i], self.train_attr_num_ids_list)]

            ood_gt_i = sum([v*np.power(2, n) for n, v in enumerate(ood_gt)])
            ood_pred_i = sum([v*np.power(2, n) for n, v in enumerate(ood_pred)])

            cmat[ood_gt_i, ood_pred_i] += 1

        # normalize matrix
        cmat_n = cmat.copy().astype(np.float)
        for ri in range(cmat.shape[0]):
            cmat_n[ri, :] = cmat[ri, :] / cmat[ri, :].sum()

        plot_path = os.path.join(self.plot_dir, 'Evaluation_Phase_{}_OE_Confusion_Matrix.jpg'.
                                 format(self.phase_name))
        tick_names = ["{0:b}".format(e).zfill(num_attrs)[::-1].replace('0', 'In').replace('1', 'Out')
                      for e in range(num_entries)]
        plotting.plot_grid_cm(cmat_n, f'Outlier Confusion Matrix (Normalized)',
                              'Prediction', 'Groundtruth', plot_path,
                              fmt='.2g', x_tick_names=tick_names, y_tick_names=tick_names)

    def plot_cross_attribute_confidence_scores(self, all_pred_conf_output, all_attr_gt):
        # get indices with single ood
        group_names = ['iid', *[f'ood_{i}' for i in range(self.num_attrs)]]
        group_indices_dict = {group_name: [] for group_name in group_names}

        for i in range(all_attr_gt.shape[0]):
            ood_flags = np.greater_equal(all_attr_gt[i].cpu(), self.train_attr_num_ids_list).cpu().numpy()
            ood_cnt = ood_flags.sum()
            if ood_cnt == 0:
                group_indices_dict['iid'].append(i)
            elif ood_cnt == 1:
                attr_id = ood_flags.argmax()
                group_indices_dict[f'ood_{attr_id}'].append(i)

        # get data matrix
        score_group_mat = np.zeros((self.num_attrs, len(group_names)))
        for i in range(score_group_mat.shape[0]):
            for j in range(score_group_mat.shape[1]):
                score_group_mat[i, j] = all_pred_conf_output[i][group_indices_dict[group_names[j]]].mean()

        plot_path = os.path.join(self.plot_dir, 'Evaluation_Phase_{}_Cross-Attribute_Confidence_Scores_Matrix.jpg'.
                                 format(self.phase_name))
        plotting.plot_grid_cm(score_group_mat, 'Confidence Score', 'Groundtruth Group', 'Attribute Index', plot_path,
                              fmt='.2g', x_tick_names=group_names)

    def refine_prediction_with_thresholds(self, all_pred_output, all_pred_conf_output, attr_score_threshold_list):
        all_pred_extended = []
        for i in range(self.num_attrs):
            attr_num_ids = self.train_attr_num_ids_list[i]

            # prediction
            pred_raw = all_pred_output[i]
            conf = all_pred_conf_output[i]
            threshold = attr_score_threshold_list[i]

            # inlier vs outlier confusion matrix
            pred_extended = pred_raw.copy()
            pred_extended[conf <= threshold] = attr_num_ids
            all_pred_extended.append(pred_extended)

        return all_pred_extended
