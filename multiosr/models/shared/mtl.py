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


class MTLLossCombinator(nn.Module):
    def __init__(self, num_tasks, c_type='uncertainty_weighting'):
        super().__init__()
        assert(c_type in ['sum', 'uncertainty_weighting'])
        self.num_tasks = num_tasks
        self.c_type = c_type

        if self.c_type == 'uncertainty_weighting':
            self.sigmas = nn.Parameter(torch.ones(num_tasks))

    def forward(self, loss_list):
        total_loss = 0

        if self.c_type == 'sum':
            total_loss = torch.sum(torch.stack(loss_list))
        elif self.c_type == 'uncertainty_weighting':
            for i, l in enumerate(loss_list):
                total_loss += 0.5 * l / (self.sigmas[i] ** 2) + torch.log(self.sigmas[i])

        return total_loss
