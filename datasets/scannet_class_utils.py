'''
Modified from SparseConvNet data preparation: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py
https://github.com/dvlab-research/PointGroup/blob/master/dataset/scannetv2/prepare_data_inst.py
'''

SCANNET_LABELS_20 = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', \
                'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']

SCANNET_LABELS_21 = ['unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', \
                 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refridgerator', 'picture', 'cabinet', \
                 'otherfurniture']


SCANNET_THING_MASK_21 = [False, 
                         False, False, True, True, True, True, True, False, False, False, 
                         False, False, True, False, False, False, True, True, False, False]

SCANNET_COLOR_MAP_20 = {
    0: (0., 0., 0.), # unlabel/unknown
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    16: (219., 219., 141.),
    24: (255., 127., 14.),
    28: (158., 218., 229.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    36: (227., 119., 194.),
    39: (82., 84., 163.), # otherfurniture
}
