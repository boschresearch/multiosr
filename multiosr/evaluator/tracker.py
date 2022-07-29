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


class StatTracker(object):
    def __init__(self):
        self.phase_val_data_dict = None

    def is_init(self):
        return self.phase_val_data_dict is not None

    def initialize(self, phase_val_names_dict):
        self.phase_val_data_dict = {phase: {name: [] for name in names} for (phase, names) in phase_val_names_dict.items()}

    def push_epoch(self, name_val_dict, phase):
        for name in self.phase_val_data_dict[phase]:
            if name in name_val_dict:
                val = float(name_val_dict[name])
                self.phase_val_data_dict[phase][name].append(val)
            else:
                self.phase_val_data_dict[phase][name].append(0)
