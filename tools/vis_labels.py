import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys 

import numpy as np
from datasets import scannet_class_utils

def visualize_labels(labels, colors, out_name, loc='lower left', ncol=10):
    n_label = len(labels)
    patches = []
    for index in range(n_label):
        label = labels[index]
        cur_color = [colors[index][0] / 255., colors[index][1] / 255., colors[index][2] / 255.]
        red_patch = mpatches.Patch(color=cur_color, label=label)
        patches.append(red_patch)
    plt.figure()
    plt.axis('off')
    legend = plt.legend(frameon=False, handles=patches, loc=loc, ncol=ncol, bbox_to_anchor=(0, -0.3), prop={'size': 5}, handlelength=0.7)
    fig = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array([-5,-5,5,5])))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(out_name, bbox_inches=bbox, dpi=300)
    plt.close()

def plot_scannet_color_bar():
    # gt 6450
    used_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20]
    labels = [scannet_class_utils.SCANNET_LABELS_20[i-1] for i in used_ids]
    colors = [[j for j in scannet_class_utils.SCANNET_COLOR_MAP_20.values()][i] for i in used_ids]

    # labels.append('unlabeled')
    # colors.append((0, 0, 0))

    labels.append('unlabeled')
    colors.append((0, 0, 0))

    out_name = './teaser_color_bar.png'
    visualize_labels(labels, colors, out_name)

plot_scannet_color_bar()