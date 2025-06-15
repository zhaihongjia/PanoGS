scenes = ['office0', 'office1', 'office2', 'office3', 'office4', 'room0', 'room1', 'room2']

valid_class_ids = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23,
                   26, 29, 31, 34, 35, 37, 40, 44, 47, 52, 54, 56, 59, 60, 61,
                   62, 63, 64, 65, 70, 71, 76, 78, 79, 80, 82, 83, 87, 88, 91,
                   92, 93, 95, 97, 98]

class_popularity = [47, 18, 22, 13,  5, 37, 40, 50, 49, 24, 21, 30,  2, 43, 11,
                    29,  4, 44, 17,  3, 46,  1, 38, 28, 23, 19, 16, 26, 36, 45,
                    32,  0, 33, 25, 31, 27, 20, 48,  6,  8, 14, 35, 42, 10, 41,
                    34,  7, 12,  9, 15, 39]
                    
num_classes = len(valid_class_ids)  # 51

map_to_reduced = {
    93:	0,
    31:	1,
    40:	2,
    20:	3,
    12:	4,
    76:	5,
    80:	6,
    98:	7,
    97:	8,
    47:	9,
    37:	10,
    61:	11,
    8:	12,
    87:	13,
    18:	14,
    60:	15,
    11:	16,
    88:	17,
    29:	18,
    10:	19,
    92:	20,
    7:	21,
    78:	22,
    59:	23,
    44:	24,
    34:	25,
    26:	26,
    54:	27,
    71:	28,
    91:	29,
    63:	30,
    3:	31,
    64:	32,
    52:	33,
    62:	34,
    56:	35,
    35:	36,
    95:	37,
    13:	38,
    15:	39,
    22:	40,
    70:	41,
    83:	42,
    17:	43,
    82:	44,
    65:	45,
    14:	46,
    19:	47,
    16:	48,
    23:	49,
    79:	50,
    -1: 51,
    -2: 51,
    256:51,
    }

SCANNET_LABELS_20 = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', \
                'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']

reduced_to_scannet_color = {
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
    14: (247., 182., 210.),
    39: (82., 84., 163.), # otherfurniture
}

MATTERPORT_COLOR_MAP_21 = {
    1: (174., 199., 232.), # wall
    2: (152., 223., 138.), # floor
    3: (31., 119., 180.), # cabinet
    4: (255., 187., 120.), # bed
    5: (188., 189., 34.), # chair
    6: (140., 86., 75.), # sofa
    7: (255., 152., 150.), # table
    8: (214., 39., 40.), # door
    9: (197., 176., 213.), # window
    10: (148., 103., 189.), # bookshelf
    11: (196., 156., 148.), # picture
    12: (23., 190., 207.), # counter
    14: (247., 182., 210.), # desk
    16: (219., 219., 141.), # curtain
    24: (255., 127., 14.), # refrigerator
    28: (158., 218., 229.), # shower curtain
    33: (44., 160., 44.), # toilet
    34: (112., 128., 144.), # sink
    36: (227., 119., 194.), # bathtub
    39: (82., 84., 163.), # other
    # 41: (186., 197., 62.), # ceiling
    41: (58., 98., 26.), # ceiling
    0: (0., 0., 0.), # unlabel/unknown
}

MATTERPORT_LABELS_21 = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                        'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'other', 
                        'ceiling']
                        
MATTERPORT_THINGMASK_21 = [False, False, True, True, True, True, True, False, False, False, 
                           True, False, True, False, False, False, True, True, False, False, \
                           False]

reduced_to_scannet_names = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'desk', 'otherfurniture']

reduced_to_mp21 = {
    0: 1, # wall
    1: 21, # ceiling: wall # zhj: no mapping in panopticlifting
    2: 2, # floor
    3: 5, # chair
    4: 9, # blinds: window
    5: 6, # sofa
    6: 7, # table
    7: 2, # rug: floor
    8: 9, # window
    9: 20, # lamp: otherfurniture
    10: 8, # door
    11: 4, # pillow: bed
    12: 7, # bench: table
    13: 20, # tv-screen: otherfurniture
    14: 3, # cabinet
    15: 1, # pillar: wall
    16: 4, # blanket: bed
    17: 20, # tv-stand: otherfurniture
    18: 20, # cushion: otherfurniture
    19: 20, # bin: otherfurniture
    20: 20, # vent: otherfurniture # zhj: no mapping in panopticlifting
    21: 4, # bed
    22: 5, # stool: chair
    23: 20, # picture: otherfurniture
    24: 20, # indoor-plant: otherfurniture
    25: 13, # desk
    26: 4, # comforter: bed
    27: 3, # nightstand: cabinet
    28: 10, # shelf: bookshelf
    29: 20, # vase: otherfurniture
    30: 20, # plant-stand: otherfurniture
    31: 20, # basket: otherfurniture
    32: 20, # plate: otherfurniture
    33: 20, # monitor: otherfurniture
    34: 20, # pipe: otherfurniture
    35: 20, # panel: otherfurniture
    36: 20, # desk-organizer: otherfurniture
    37: 1, # wall-plug: wall
    38: 20, # book: otherfurniture
    39: 20, # box: otherfurniture
    40: 20, # clock: otherfurniture
    41: 20, # sculpture: otherfurniture
    42: 20, # tissue-paper: otherfurniture
    43: 20, # camera: otherfurniture
    44: 20, # tablet: otherfurniture
    45: 20, # pot: otherfurniture
    46: 20, # bottle: otherfurniture
    47: 20, # candle: otherfurniture
    48: 20, # bowl: otherfurniture
    49: 20, # cloth: otherfurniture # zhj: no mapping in panopticlifting
    50: 1, # switch: wall
    51: 20, # otherfurniture
}

reduced_to_scannet_to_continues = {
    1: 0, # wall
    2: 1, # floor
    3: 2, # cabinet
    4: 3, # bed
    5: 4, # chair
    6: 5, # sofa
    7: 6, # table
    8: 7, # door
    9: 8, # window
    10: 9, # bookshelf
    13: 10, # desk
    20: 11, # otherfurniture
}

reduced_to_scannet = {
    0: 1, # wall
    1: 1, # ceiling: wall # zhj: no mapping in panopticlifting
    2: 2, # floor
    3: 5, # chair
    4: 9, # blinds: window
    5: 6, # sofa
    6: 7, # table
    7: 2, # rug: floor
    8: 9, # window
    9: 20, # lamp: otherfurniture
    10: 8, # door
    11: 4, # pillow: bed
    12: 7, # bench: table
    13: 20, # tv-screen: otherfurniture
    14: 3, # cabinet
    15: 1, # pillar: wall
    16: 4, # blanket: bed
    17: 20, # tv-stand: otherfurniture
    18: 20, # cushion: otherfurniture
    19: 20, # bin: otherfurniture
    20: 20, # vent: otherfurniture # zhj: no mapping in panopticlifting
    21: 4, # bed
    22: 5, # stool: chair
    23: 20, # picture: otherfurniture
    24: 20, # indoor-plant: otherfurniture
    25: 13, # desk
    26: 4, # comforter: bed
    27: 3, # nightstand: cabinet
    28: 10, # shelf: bookshelf
    29: 20, # vase: otherfurniture
    30: 20, # plant-stand: otherfurniture
    31: 20, # basket: otherfurniture
    32: 20, # plate: otherfurniture
    33: 20, # monitor: otherfurniture
    34: 20, # pipe: otherfurniture
    35: 20, # panel: otherfurniture
    36: 20, # desk-organizer: otherfurniture
    37: 1, # wall-plug: wall
    38: 20, # book: otherfurniture
    39: 20, # box: otherfurniture
    40: 20, # clock: otherfurniture
    41: 20, # sculpture: otherfurniture
    42: 20, # tissue-paper: otherfurniture
    43: 20, # camera: otherfurniture
    44: 20, # tablet: otherfurniture
    45: 20, # pot: otherfurniture
    46: 20, # bottle: otherfurniture
    47: 20, # candle: otherfurniture
    48: 20, # bowl: otherfurniture
    49: 20, # cloth: otherfurniture # zhj: no mapping in panopticlifting
    50: 1, # switch: wall
    51: 20, # otherfurniture
}

class_names_reduced = [
    'wall',
    'ceiling',
    'floor',
    'chair',
    'blinds',
    'sofa',
    'table',
    'rug',
    'window',
    'lamp',
    'door',
    'pillow',
    'bench',
    'tv-screen',
    'cabinet',
    'pillar',
    'blanket',
    'tv-stand',
    'cushion',
    'bin',
    'vent',
    'bed',
    'stool',
    'picture',
    'indoor-plant',
    'desk',
    'comforter',
    'nightstand',
    'shelf',
    'vase',
    'plant-stand',
    'basket',
    'plate',
    'monitor',
    'pipe',
    'panel',
    'desk-organizer',
    'wall-plug',
    'book',
    'box',
    'clock',
    'sculpture',
    'tissue-paper',
    'camera',
    'tablet',
    'pot',
    'bottle',
    'candle',
    'bowl',
    'cloth', # zhj: no exist in replica_to_scannet_reduced
    'switch',
    ]

class_names = [
    '0',                # 0
    'backpack',         # 1
    'base-cabinet',     # 2
    'basket',           # 3
    'bathtub',          # 4
    'beam',
    'beanbag',
    'bed',
    'bench',
    'bike',
    'bin',
    'blanket',
    'blinds',
    'book',
    'bottle',
    'box',
    'bowl',
    'camera',
    'cabinet',
    'candle',
    'chair',
    'chopping-board',
    'clock',
    'cloth',
    'clothing',
    'coaster',
    'comforter',
    'computer-keyboard',
    'cup',
    'cushion',
    'curtain',
    'ceiling',
    'cooktop',
    'countertop',
    'desk',
    'desk-organizer',
    'desktop-computer',
    'door',
    'exercise-ball',
    'faucet',
    'floor',
    'handbag',
    'hair-dryer',
    'handrail',
    'indoor-plant',
    'knife-block',
    'kitchen-utensil',
    'lamp',
    'laptop',
    'major-appliance',
    'mat',
    'microwave',
    'monitor',
    'mouse',
    'nightstand',
    'pan',
    'panel',
    'paper-towel',
    'phone',
    'picture',
    'pillar',
    'pillow',
    'pipe',
    'plant-stand',
    'plate',
    'pot',
    'rack',
    'refrigerator',
    'remote-control',
    'scarf',
    'sculpture',
    'shelf',
    'shoe',
    'shower-stall',
    'sink',
    'small-appliance',
    'sofa',
    'stair',
    'stool',
    'switch',
    'table',
    'table-runner',
    'tablet',
    'tissue-paper',
    'toilet',
    'toothbrush',
    'towel',
    'tv-screen',
    'tv-stand',
    'umbrella',
    'utensil-holder',
    'vase',
    'vent',
    'wall',
    'wall-cabinet',
    'wall-plug',
    'wardrobe',
    'window',
    'rug',
    'logo',
    'bag',
    'set-of-clothing',
]

SCANNET_COLOR_MAP_200 = {
    0: (196., 51., 182.),   # 0
    1: (174., 199., 232.),  # 1
    2: (188., 189., 34.),   # 2
    3: (152., 223., 138.),  # 3
    4: (255., 152., 150.),  # 4
    5: (214., 39., 40.),    # 5
    6: (91., 135., 229.),   # 6
    7: (31., 119., 180.),   # 7
    8: (229., 91., 104.),   # 8
    9: (247., 182., 210.),  # 9
    10: (91., 229., 110.),  # 10
    11: (255., 187., 120.), # 11
    13: (141., 91., 229.),  # 12
    14: (112., 128., 144.), # 13
    15: (196., 156., 148.), # 14
    16: (197., 176., 213.), # 15
    17: (44., 160., 44.),   # 16
    18: (148., 103., 189.), # 17
    19: (229., 91., 223.),  # 18
    21: (219., 219., 141.), # 19
    22: (192., 229., 91.),  # 20
    23: (88., 218., 137.),  # 21
    24: (58., 98., 137.),   # 22
    26: (177., 82., 239.),  # 23
    27: (255., 127., 14.),  # 24
    28: (237., 204., 37.),  # 25
    29: (41., 206., 32.),   # 26
    31: (62., 143., 148.),  # 27
    32: (34., 14., 130.),   # 28
    33: (143., 45., 115.),  # 29
    34: (137., 63., 14.),   # 30
    35: (23., 190., 207.),  # 31
    36: (16., 212., 139.),  # 32
    38: (90., 119., 201.),  # 33
    39: (125., 30., 141.),  # 34
    40: (150., 53., 56.),   # 35
    41: (186., 197., 62.),  # 36
    42: (227., 119., 194.), # 37
    44: (38., 100., 128.),  # 38
    45: (120., 31., 243.),  # 39
    46: (154., 59., 103.),  # 40
    47: (169., 137., 78.),  # 41
    48: (143., 245., 111.), # 42
    49: (37., 230., 205.),  # 43
    50: (14., 16., 155.),   # 44
    51: (208., 49., 84.),   # 45
    52: (237., 80., 38.),   # 46
    54: (138., 175., 62.),  # 47
    55: (158., 218., 229.), # 48
    56: (38., 96., 167.),   # 49
    57: (190., 77., 246.),  # 50
    58: (0., 0., 0.),       # 51
    59: (208., 193., 72.),
    62: (55., 220., 57.),
    63: (10., 125., 140.),
    64: (76., 38., 202.),
    65: (191., 28., 135.),
    66: (211., 120., 42.),
    67: (118., 174., 76.),
    68: (17., 242., 171.),
    69: (20., 65., 247.),
    70: (208., 61., 222.),
    71: (162., 62., 60.),
    72: (210., 235., 62.),
    73: (45., 152., 72.),
    74: (35., 107., 149.),
    75: (160., 89., 237.),
    76: (227., 56., 125.),
    77: (169., 143., 81.),
    78: (42., 143., 20.),
    79: (25., 160., 151.),
    80: (82., 75., 227.),
    82: (253., 59., 222.),
    84: (240., 130., 89.),
    86: (123., 172., 47.),
    87: (71., 194., 133.),
    88: (24., 94., 205.),
    89: (134., 16., 179.),
    90: (159., 32., 52.),
    93: (213., 208., 88.),
    95: (64., 158., 70.),
    96: (18., 163., 194.),
    97: (65., 29., 153.),
    98: (177., 10., 109.),
    99: (152., 83., 7.),
    100: (83., 175., 30.),
    101: (18., 199., 153.),
    102: (61., 81., 208.),
    103: (213., 85., 216.),
    104: (170., 53., 42.),
    105: (161., 192., 38.),
    106: (23., 241., 91.),
    107: (12., 103., 170.),
    110: (151., 41., 245.),
    112: (133., 51., 80.),
    115: (184., 162., 91.),
    116: (50., 138., 38.),
    118: (31., 237., 236.),
    120: (39., 19., 208.),
    121: (223., 27., 180.),
    122: (254., 141., 85.),
    125: (97., 144., 39.),
    128: (106., 231., 176.),
    130: (12., 61., 162.),
    131: (124., 66., 140.),
    132: (137., 66., 73.),
    134: (250., 253., 26.),
    136: (55., 191., 73.),
    138: (60., 126., 146.),
    139: (153., 108., 234.),
    140: (184., 58., 125.),
    141: (135., 84., 14.),
    145: (139., 248., 91.),
    148: (53., 200., 172.),
    154: (63., 69., 134.),
    155: (190., 75., 186.),
    156: (127., 63., 52.),
    157: (141., 182., 25.),
    159: (56., 144., 89.),
    161: (64., 160., 250.),
    163: (182., 86., 245.),
    165: (139., 18., 53.),
    166: (134., 120., 54.),
    168: (49., 165., 42.),
    169: (51., 128., 133.),
    170: (44., 21., 163.),
    177: (232., 93., 193.),
    180: (176., 102., 54.),
    185: (116., 217., 17.),
    188: (54., 209., 150.),
    191: (60., 99., 204.),
    193: (129., 43., 144.),
    195: (252., 100., 106.),
    202: (187., 196., 73.),
    208: (13., 158., 40.),
    213: (52., 122., 152.),
    214: (128., 76., 202.),
    221: (187., 50., 115.),
    229: (180., 141., 71.),
    230: (77., 208., 35.),
    232: (72., 183., 168.),
    233: (97., 99., 203.),
    242: (172., 22., 158.),
    250: (155., 64., 40.),
    261: (118., 159., 30.),
    264: (69., 252., 148.),
    276: (45., 103., 173.),
    283: (111., 38., 149.),
    286: (184., 9., 49.),
    300: (188., 174., 67.),
    304: (53., 206., 53.),
    312: (97., 235., 252.),
    323: (66., 32., 182.),
    325: (236., 114., 195.),
    331: (241., 154., 83.),
    342: (133., 240., 52.),
    356: (16., 205., 144.),
    370: (75., 101., 198.),
    392: (237., 95., 251.),
    395: (191., 52., 49.),
    399: (227., 254., 54.),
    408: (49., 206., 87.),
    417: (48., 113., 150.),
    488: (125., 73., 182.),
    540: (229., 32., 114.),
    562: (158., 119., 28.),
    570: (60., 205., 27.),
    572: (18., 215., 201.),
    581: (79., 76., 153.),
    609: (134., 13., 116.),
    748: (192., 97., 63.),
    776: (108., 163., 18.),
    1156: (95., 220., 156.),
    1163: (98., 141., 208.),
    1164: (144., 19., 193.),
    1165: (166., 36., 57.),
    1166: (212., 202., 34.),
    1167: (23., 206., 34.),
    1168: (91., 211., 236.),
    1169: (79., 55., 137.),
    1170: (182., 19., 117.),
    1171: (134., 76., 14.),
    1172: (87., 185., 28.),
    1173: (82., 224., 187.),
    1174: (92., 110., 214.),
    1175: (168., 80., 171.),
    1176: (197., 63., 51.),
    1178: (175., 199., 77.),
    1179: (62., 180., 98.),
    1180: (8., 91., 150.),
    1181: (77., 15., 130.),
    1182: (154., 65., 96.),
    1183: (197., 152., 11.),
    1184: (59., 155., 45.),
    1185: (12., 147., 145.),
    1186: (54., 35., 219.),
    1187: (210., 73., 181.),
    1188: (221., 124., 77.),
    1189: (149., 214., 66.),
    1190: (72., 185., 134.),
    1191: (42., 94., 198.),
    }