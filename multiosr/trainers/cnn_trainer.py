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
import argparse
import logging
from tqdm import tqdm
import torch.optim as optim
import numpy as np

from .base_trainer import BaseTrainer
from multiosr.models.cnn.models import CNNModel
from multiosr.utils.training_tools import build_dataloader, build_optimizer
from multiosr.utils.misc import MeanAccumulatorSet, get_iid_sample_indices, batch_wise_to_sample_wise_data, save_obj


class CNN(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

        # create model
        model_params = self.model_spec['params']
        model_args = argparse.Namespace(**model_params['args'])
        self.model = CNNModel(self.train_attr_num_ids_list, model_args).to(self.dev)

        # create optimizer
        self.optimizer = build_optimizer(self.learning_rule['optimizer'], self.model.parameters())

        # create scheduler
        self.scheduler = None
        if 'lr_decay_step' in self.learning_rule:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.learning_rule['lr_decay_step'],
                                                       gamma=0.1)

        # set validation criteria
        self.set_validation_criteria('loss_val', np.min)

    def train_epoch(self, epoch):
        self.model.train()

        # log
        logging.info('Using lr={}'.format(self.optimizer.param_groups[0]['lr']))

        dataset = self.dataset_catalog['target_train'][0]
        dataloader = build_dataloader(self.learning_rule['dataloader_spec']['params'], dataset, shuffle=True)

        mean_accumulators = MeanAccumulatorSet()
        for idx, data in tqdm(enumerate(dataloader), desc="Training...", total=len(dataloader)):
            # convert to cuda
            data = [d.to(self.dev) for d in data]
            loss, _, aux_dict = self.model.forward(data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if idx == 0:
                mean_accumulators.reset_name_accumulator_dict({'loss_train', *aux_dict.keys()})
            mean_accumulators.accumulate({'loss_train': loss.item(),
                                          **aux_dict}, data[0].shape[0])

        log_dict = mean_accumulators.get_name_mean_dict()
        logging.info('Epoch: {} | Loss: {}'.format(epoch, log_dict['loss_train']))

        if self.scheduler is not None:
            self.scheduler.step()

        return log_dict

    @torch.no_grad()
    def validate_epoch(self, epoch):
        del epoch
        self.model.eval()

        # initialize validation
        dataset_train = self.dataset_catalog['target_train'][0]
        dataloader_train = build_dataloader(self.learning_rule['dataloader_spec']['params'], dataset_train, shuffle=True)

        # dataset
        ds_attribute_index = self.dataset_catalog['target_val'][1]['ds_attribute_index']

        dataset = self.dataset_catalog['target_val'][0]
        dataloader = build_dataloader(self.learning_rule['dataloader_spec']['params'], dataset, shuffle=False)

        all_pred, all_pred_conf, all_attr_gt = [], [], []

        mean_accumulators = MeanAccumulatorSet()
        for idx, data in enumerate(dataloader):
            # convert to cuda
            data = [d.to(self.dev) for d in data]
            loss, (pred, pred_conf), aux_dict = self.model.val_forward(data)

            # log
            attr_gt = data[1]
            all_pred.append(pred)
            all_pred_conf.append(pred_conf)
            all_attr_gt.append(attr_gt)

            if idx == 0:
                mean_accumulators.reset_name_accumulator_dict({'loss_val', *aux_dict.keys()})
            mean_accumulators.accumulate(aux_dict, data[0].shape[0])

            num_iids = len(get_iid_sample_indices(data[ds_attribute_index], self.train_attr_num_ids_list))
            if num_iids > 0:
                mean_accumulators.accumulate({'loss_val': loss.item() if loss is not None else 0}, num_iids)

        # reorganized output from batch-wise to sample-wise
        all_attr_gt = torch.cat(all_attr_gt) # (N x num_attrs)
        all_pred_output = batch_wise_to_sample_wise_data(all_pred) # [(N x num_ids), ...<num_attrs>]
        all_pred_conf_output = batch_wise_to_sample_wise_data(all_pred_conf) # [(N), ...<num_attrs>]

        log_dict = mean_accumulators.get_name_mean_dict()
        return (all_pred_output, all_pred_conf_output, all_attr_gt), log_dict

    @torch.no_grad()
    def test(self):
        self.model.eval()

        # initialize validation
        dataset_train = self.dataset_catalog['target_train'][0]

        # dataset
        ds_attribute_index = self.dataset_catalog['target_val'][1]['ds_attribute_index']

        dataset = self.dataset_catalog['target_test'][0]
        dataloader = build_dataloader(self.learning_rule['dataloader_spec']['params'], dataset, shuffle=True)

        all_pred, all_pred_conf, all_attr_gt = [], [], []

        mean_accumulators = MeanAccumulatorSet()
        for idx, data in enumerate(dataloader):
            # convert to cuda
            data = [d.to(self.dev) for d in data]
            loss, (pred, pred_conf), aux_dict = self.model.val_forward(data)

            # log
            attr_gt = data[1]
            all_pred.append(pred)
            all_pred_conf.append(pred_conf)
            all_attr_gt.append(attr_gt)

            if idx == 0:
                mean_accumulators.reset_name_accumulator_dict({'loss_val', *aux_dict.keys()})
            mean_accumulators.accumulate({**aux_dict}, data[0].shape[0])

            num_iids = len(get_iid_sample_indices(data[ds_attribute_index], self.train_attr_num_ids_list))
            if num_iids > 0:
                mean_accumulators.accumulate({'loss_val': loss.item() if loss is not None else 0}, num_iids)

        # reorganized output from batch-wise to sample-wise
        all_attr_gt = torch.cat(all_attr_gt)
        all_pred_output = batch_wise_to_sample_wise_data(all_pred)
        all_pred_conf_output = batch_wise_to_sample_wise_data(all_pred_conf)

        log_dict = mean_accumulators.get_name_mean_dict()

        return (all_pred_output, all_pred_conf_output, all_attr_gt), log_dict

