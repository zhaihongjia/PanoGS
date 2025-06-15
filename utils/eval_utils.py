import numpy as np

def calculate_iou_3d(preds, targets, num_classes):
    iou_per_class = np.zeros(num_classes)
    acc_per_class = np.zeros(num_classes)
    mask_per_class = np.zeros(num_classes, dtype=bool)
    
    for c in range(num_classes):
        pred_class = (preds == c)
        target_class = (targets == c)
        
        intersection = (pred_class & target_class).sum().astype(float)
        union = (pred_class | target_class).sum().astype(float)
        
        if union != 0:
            iou_per_class[c] = intersection / union
        if target_class.sum().astype(float) != 0:
            acc_per_class[c] = intersection / target_class.sum().astype(float)
        
        if target_class.sum() != 0:
            mask_per_class[c] = True

    return iou_per_class, acc_per_class, mask_per_class 
