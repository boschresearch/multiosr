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
from torch import nn
import torchvision


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.05)
        if m.bias is not None:
            m.bias.data.fill_(0)
 

def get_feature_extractor(feature_type, latent_dim):
    if feature_type == 'resnet18':
        feature_extractor_raw, feature_dim_raw = create_feature_extractor_resnet18()
        feature_extractor = nn.Sequential(feature_extractor_raw, nn.Linear(feature_dim_raw, latent_dim))
    elif feature_type == 'lenet':
        feature_extractor = LeNetCustom(latent_dim, bn_type='none')
        feature_extractor.apply(weights_init)
    else:
        raise RuntimeError('feature_type {} is currently not supported.'.format(feature_type))

    return feature_extractor


def create_feature_extractor_resnet18():
    # get resnet without FCN
    resnet_full = torchvision.models.resnet18(pretrained=True)
    resnet_without_fcn = list(resnet_full.children())[:-1]
    resnet = nn.Sequential(*resnet_without_fcn, nn.Flatten())

    return resnet, 512


def create_feature_extractor_lenet(latent_dim):
    return LeNetCustom(latent_dim, bn_type='bn')


def get_activation(type):
    assert(type in ['relu', 'leaky_relu'])
    if 'relu':
        return nn.ReLU()
    else:
        return nn.LeakyReLU(0.2)


class LeNetCustom(nn.Module):
    def __init__(self, latent_dim=84, bn_type='none', dropout=False, activation_type='relu') -> None:
        super(LeNetCustom, self).__init__()
        assert(bn_type in ['none', 'bn'])
        self.latent_dim = latent_dim
        self.bn_type = bn_type

        self.dr1 = nn.Dropout2d(0.2) if dropout else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        if self.bn_type == 'none':
            self.bn1 = nn.Identity()
        elif bn_type == 'bn':
            self.bn1 = nn.BatchNorm2d(6)
        self.activation1 = get_activation(activation_type)

        self.dr2 = nn.Dropout2d(0.2) if dropout else nn.Identity()
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        if bn_type == 'none':
            self.bn2 = nn.Identity()
        elif bn_type == 'bn':
            self.bn2 = nn.BatchNorm2d(16)
        self.avgpool = nn.AdaptiveAvgPool2d((14, 14))
        self.activation2 = get_activation(activation_type)

        self.fc3 = nn.Linear(16 * 14 * 14, 120)
        self.activation3 = get_activation(activation_type)

        self.fc4 = nn.Linear(120, latent_dim)
        self.activation4 = get_activation(activation_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1
        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)

        # 2
        x = self.dr2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # 3
        x = self.fc3(x)
        x = self.activation3(x)

        # 4
        x = self.fc4(x)
        x = self.activation4(x)

        return x
