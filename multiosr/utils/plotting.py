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

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn')
matplotlib.use('Agg')
import seaborn as sb


def plot_grid_cm(grid_data, title, x_label, y_label, savepath, cmap='rocket_r', fmt='.2g',
                 x_tick_names=None, y_tick_names=None, vmin=None, vmax=None, figsize=None,
                 x_rotation=45, y_rotation='horizontal', x_ha='center', heatmap_fontsize=14,
                 x_tick_fontsize=7, y_tick_fontsize=7):
    classes_y = [str(i) for i in range(grid_data.shape[0])] if y_tick_names is None else y_tick_names
    classes_x = [str(i) for i in range(grid_data.shape[1])] if x_tick_names is None else x_tick_names
    fig = plt.figure(figsize=((5, 1 + 0.5 * len(classes_y)) if figsize is None else figsize))
    ax = fig.add_subplot(111)
    sb.heatmap(grid_data, cmap=plt.get_cmap(cmap), annot=True, fmt=fmt, vmin=vmin, vmax=vmax,
               annot_kws={"size": heatmap_fontsize})
    ax.set_xticks([i + 0.5 for i in range(len(classes_x))])
    ax.set_yticks([i + 0.5 for i in range(len(classes_y))])
    ax.set_xticklabels(classes_x, rotation=x_rotation, ha=x_ha, fontsize=x_tick_fontsize)
    ax.set_yticklabels(classes_y, rotation=y_rotation, fontsize=y_tick_fontsize)
    plt.title(title)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close('all')


def plot_XY_curve(X, Y, xlabel, ylabel, filename):
    plt.figure(figsize=(5, 3))
    plt.plot(X, Y, 'x-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close('all')
