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
import sys
from datetime import datetime


def run_job(method, dataset, training_seed):
    # arguments
    repo_dir = os.getcwd()
    dataset_str = dataset.replace('/', '-')
    method_str = method.replace('/', '-')
    result_dir = f'{repo_dir}/outputs/study_{dataset_str}_{method_str}' \
                 + '-' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    dataset_spec = f'{repo_dir}/configs/datasets/{dataset}.json'
    method_spec = f'{repo_dir}/configs/methods/{method}/method_spec.json'

    # run command (Note: It is suggested to adjust this command execution to perform on a cluster)
    exe = 'multiosr.train'
    command_str = f'python -m {exe} --result_dir {result_dir} --dataset_spec {dataset_spec} ' \
                  f'--method_spec {method_spec} --test --training_seed {training_seed}'
    print(f'Executing {command_str}')
    sys.stdout.flush()
    os.system(command_str)


if __name__ == '__main__':
    # study configurations (Note: These configurations can be modified to run only what you need)
    training_seed_list = [2000]
    dataset_list = ['color_mnist/uncorrelated', 'color_mnist/semi_correlated', 'color_mnist/correlated',
                    'ut_zappos/ut_zappos']

    dataset_to_method_list = {
        'color_mnist/uncorrelated': ['lenet/MSP'],
        'color_mnist/semi_correlated': ['lenet/MSP'],
        'color_mnist/correlated': ['lenet/MSP'],
        'ut_zappos/ut_zappos': ['resnet18/MSP']
    }

    # run all specified studies
    for training_seed in training_seed_list:
        for dataset in dataset_list:
            method_list = dataset_to_method_list[dataset]
            for method in method_list:
                run_job(method, dataset, training_seed)
