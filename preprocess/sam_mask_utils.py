import numpy as np
import copy

def get_single_mask(annotations, metric='predicted_iou', descending=False):
    # metric: predicted_iou or area
    sorted_masks = sorted(annotations, key=(lambda x: x['predicted_iou']), reverse=descending) # True: descending, False: ascending
    # print('sorted_masks: ', len(sorted_masks))
    mask = np.full((sorted_masks[0]['segmentation'].shape[0], sorted_masks[0]['segmentation'].shape[1]), -1, dtype=int)
    
    mask_id = 0
    for ann in sorted_masks:
        m = ann['segmentation']
        mask_id += 1
        mask[m] = mask_id
    
    # start from 1, 0 is invalid
    mask = num_to_natural(mask) + 1
    return mask

def num_to_natural(mask):
    '''
    Change the group number to natural number arrangement (code credit: SAM3D)
    '''
    if np.all(mask == -1):
        return mask
    array = copy.deepcopy(mask).astype(int)

    unique_values = np.unique(array[array != -1])
    mapping = np.full(np.max(unique_values) + 2, -1)
    mapping[unique_values + 1] = np.arange(len(unique_values))  # map ith(start from 0) group_id to i
    array = mapping[array + 1]
    return array

def viz_mask(mask):
    array = np.zeros(tuple(mask.shape) + (3,))
    if np.all(mask == -1):
        return array
    unique_values = np.unique(mask[mask != -1])
    for i in unique_values:
        array[mask == i] = np.random.random((3))

    return array * 255
