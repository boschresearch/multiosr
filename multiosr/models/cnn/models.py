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
import torch.nn as nn
import torch.nn.functional as F

import multiosr.models.shared.arch as arch
import multiosr.models.shared.mtl as mtl
from multiosr.utils.misc import get_iid_sample_indices
from . import feature_extractors as fe


class CNNModel(nn.Module):
    def __init__(self, attr_num_ids_list, args):
        super().__init__()
        assert(args.score_type in ['softmax', 'logit'])

        self.attr_num_ids_list = attr_num_ids_list
        self.args = args
        self.num_attrs = len(self.attr_num_ids_list)
        self.encoder_ind = args.encoder_ind

        # arguments
        self.latent_dim = args.latent_dim
        self.score_type = args.score_type

        # encoder
        if self.encoder_ind:
            self.encoder = nn.ModuleList([fe.get_feature_extractor(args.encoder_type, self.latent_dim)
                                          for _ in range(self.num_attrs)])
        else:
            self.encoder = fe.get_feature_extractor(args.encoder_type, self.latent_dim)

        # attribute classifiers
        self.attr_classifier_list = nn.ModuleList(
            [arch.MLP(self.latent_dim, num_ids, 2*[self.latent_dim]) for num_ids in self.attr_num_ids_list]
        )

        # loss combinator
        self.loss_combinator = mtl.MTLLossCombinator(self.num_attrs, c_type="sum")

    def train_forward(self, x):
        imgs, attrs = x

        # predict
        if self.encoder_ind:
            attr_logit_list = [classifier(enc(imgs)) for enc, classifier in zip(self.encoder, self.attr_classifier_list)]
        else:
            z = self.encoder(imgs)
            attr_logit_list = [classifier(z) for classifier in self.attr_classifier_list]

        # loss
        loss_data_list = []
        for i in range(self.num_attrs):
            loss_data_i = F.cross_entropy(attr_logit_list[i], attrs[:, i])
            loss_data_list.append(loss_data_i) 
        loss_data = self.loss_combinator(loss_data_list)
        loss_data /= self.num_attrs 

        # total loss
        loss = loss_data

        aux_dict = {'loss_data': loss_data.item()}

        # attribute accuracy
        for i in range(self.num_attrs):
            acc = (attr_logit_list[i].argmax(dim=1) == attrs[:, i]).float().mean().item()
            aux_dict[f'attr_{i}_acc'] = acc

        return loss, attr_logit_list, aux_dict

    def val_forward(self, x):
        device = next(self.parameters()).device
        imgs, attrs = x

        # predict
        if self.encoder_ind:
            attr_logit_list = [classifier(enc(imgs)) for enc, classifier in zip(self.encoder, self.attr_classifier_list)]
        else:
            z = self.encoder(imgs)
            attr_logit_list = [classifier(z) for classifier in self.attr_classifier_list]

        # iid only loss (to preserve OSR assumption)
        iid_sample_indices = self._get_iid_sample_indices(attrs)
        loss_data = torch.tensor(0.0).to(device)
        if len(iid_sample_indices) > 0:
            loss_data_list = []
            for i in range(self.num_attrs):
                loss_data_i = F.cross_entropy(attr_logit_list[i][iid_sample_indices], attrs[iid_sample_indices, i])
                loss_data_list.append(loss_data_i) 
            loss_data = self.loss_combinator(loss_data_list)
            loss_data /= self.num_attrs 

        # total loss
        loss = loss_data

        aux_dict = {'loss_data': loss_data.item() if loss_data is not None else None}

        # attribute accuracy
        for i in range(self.num_attrs):
            acc = (attr_logit_list[i].argmax(dim=1) == attrs[:, i]).float().mean().item()
            aux_dict[f'attr_{i}_acc'] = acc

        # confidence scores
        if self.score_type == 'logit':
            attr_pred_confident_list = [logit.max(dim=1)[0] for logit in attr_logit_list]
        else:
            attr_pred_confident_list = [F.softmax(logit, dim=1).max(dim=1)[0] for logit in attr_logit_list]

        # logit
        attr_logit_list_output = attr_logit_list
        if self.score_type == 'softmax':
            for i in range(len(attr_logit_list_output)):
                attr_logit_list_output[i] = F.softmax(attr_logit_list_output[i], dim=1)

        # pair accuracy
        return loss, (attr_logit_list_output, attr_pred_confident_list), aux_dict

    def forward(self, x):
        if self.training:
            loss, pred, aux_dict = self.train_forward(x)
        else:
            with torch.no_grad():
                if isinstance(x, tuple):
                    loss, pred, aux_dict = self.train_forward(x)
                else:
                    loss, pred, aux_dict = self.val_forward(x)
        return loss, pred, aux_dict

    def _get_iid_sample_indices(self, attrs_batch):
        return get_iid_sample_indices(attrs_batch, self.attr_num_ids_list)
