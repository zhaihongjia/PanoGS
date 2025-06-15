import numpy as np
import os
import sys
from argparse import ArgumentParser

import datasets.replica_class_utils as replica_class_utils
from utils.eval_utils import calculate_iou_3d

def panoptic_evaluate_replica_scene(pred_semantics, pred_instances, gt_semantics, gt_instances, thing=False):
    metrics = {}
    thing_mask = replica_class_utils.MATTERPORT_THINGMASK_21

    TP, FP, FN = 0, 0, 0
    PQ, SQ, RQ = 0.0, 0.0, 0.0
    iou_sum = 0.0
    num_valid_class = 0.0
    AP = 0.0

    # diffrerent semantic classes
    for i in range(len(thing_mask)):
        if i == 19:
            continue
        
        # eval thing PQ
        if thing:
            if not thing_mask[i-1]:
                continue
        # eval stuff PQ
        else:
            if thing_mask[i-1]:
                continue
        
        cls_label = i
        label_name = replica_class_utils.MATTERPORT_LABELS_21[i]

        # pointclouds of current semantic class 
        pred_mask = (pred_semantics == cls_label)
        gt_mask = (gt_semantics == cls_label)

        # instances of current semantic class 
        pred_instances_cls = pred_instances[pred_mask]
        gt_instances_cls = gt_instances[gt_mask]
                
        unique_gt_instances = np.unique(gt_instances_cls)
        unique_pred_instances = np.unique(pred_instances_cls)
        
        for gt_inst in unique_gt_instances:
            # instance mask: semantic_label & instance_label
            gt_inst_mask = gt_mask & (gt_instances == gt_inst)
            max_iou = 0.0
            matched_pred_inst = -1

            if not (np.sum(gt_inst_mask) > 0.0):
                continue
            
            # find maximum miou
            for pred_inst in unique_pred_instances:
                pred_inst_mask = pred_mask & (pred_instances == pred_inst)
                intersection = np.sum(gt_inst_mask & pred_inst_mask)
                union = np.sum(gt_inst_mask | pred_inst_mask)
                iou = intersection / union if union > 0 else 0
                
                if iou > max_iou:
                    max_iou = iou
                    matched_pred_inst = pred_inst
            
            if max_iou > 0.25: 
                TP += 1
                iou_sum += max_iou
                unique_pred_instances = unique_pred_instances[unique_pred_instances != matched_pred_inst]
            else:
                FN += 1

        # False Positive: unmatched predictions
        FP = len(unique_pred_instances)  
        
        # cal. PQ, SQ, RQ
        PQ += iou_sum / (TP + 0.5 * FP + 0.5 * FN) if (TP + FP + FN) > 0 else 0
        SQ += iou_sum / TP if TP > 0 else 0
        RQ += TP / (TP + 0.5 * FP + 0.5 * FN) if (TP + FP + FN) > 0 else 0
        AP += iou_sum / (TP + FP) if (TP + FP) > 0 else 0
        num_valid_class += 1 if (TP + FP + FN) > 0 else 0
    
    metrics = {'PQ': PQ / num_valid_class, 'SQ': SQ / num_valid_class, 'RQ': RQ / num_valid_class, 'AP': AP / num_valid_class}

    return metrics

def evaluate_replica(args):
    scenes = ['room0', 'room1', 'room2', 'office0', 'office1', 'office2', 'office3', 'office4']
    
    # 19-th: others
    num_classes = 21
    mious, maccs, thing_pqs, thing_sqs, thing_rqs, stuff_pqs, stuff_sqs, stuff_rqs = [], [], [], [], [], [], [], []
    thing_aps, stuff_aps = [], []

    for scene in scenes:
        pre_sem = np.load(f'{args.pred_path}/{scene}/pre_semantic.npy')
        pre_ins = np.load(f'{args.pred_path}/{scene}/segmentation/final_seg.npy')
        gt_sem = np.load(f'{args.gt_path}/{scene}/semantic_labels_mp21.npy')
        gt_ins = np.load(f'{args.gt_path}/{scene}/instance_labels_mp21.npy')

        ious, accs, masks = calculate_iou_3d(pre_sem, gt_sem, num_classes)
        
        # not eval others
        masks[19] = False
        miou = ious[masks].mean()
        macc = accs[masks].mean()
        print(f'{scene}: miou: {miou} macc:{macc}')
        mious.append(miou)
        maccs.append(macc)
        
        # eval Thing
        metrics = panoptic_evaluate_replica_scene(pre_sem, pre_ins, gt_sem, gt_ins, thing=True)
        print(f"{scene}: Thing Performance: PQ: {metrics['PQ']} SQ: {metrics['SQ']} RQ: {metrics['RQ']}")
        thing_pqs.append(metrics['PQ'])
        thing_sqs.append(metrics['SQ'])
        thing_rqs.append(metrics['RQ'])
        thing_aps.append(metrics['AP'])

        # eval Stuff
        metrics = panoptic_evaluate_replica_scene(pre_sem, pre_ins, gt_sem, gt_ins, thing=False)
        print(f"{scene}: Stuff Performance: PQ: {metrics['PQ']} SQ: {metrics['SQ']} RQ: {metrics['RQ']}\n\n")
        stuff_pqs.append(metrics['PQ'])
        stuff_sqs.append(metrics['SQ'])
        stuff_rqs.append(metrics['RQ'])
        stuff_aps.append(metrics['AP'])

    print('Scene means: ')
    print(f'AP: {(np.array(thing_aps).mean()+np.array(stuff_aps).mean())/2} Thing AP: {np.array(thing_aps).mean()} Stuff AP: {np.array(stuff_aps).mean()}') 
    print(f'mIoU: {np.array(mious).mean()} mAcc: {np.array(maccs).mean()}') 
    print(f'thing pq: {np.array(thing_pqs).mean()} thing sq: {np.array(thing_sqs).mean()} thing rq: {np.array(thing_rqs).mean()}')
    print(f'stuff pq: {np.array(stuff_pqs).mean()} stuff sq: {np.array(stuff_sqs).mean()} stuff rq: {np.array(stuff_rqs).mean()}')
    return 

if __name__ == '__main__':
    parser = ArgumentParser(description="Evaluation.")
    parser.add_argument("--pred_path", type=str, default='/mnt/nas_10/group/hongjia/PanopticGS/Replica')
    # parser.add_argument("--pred_path", type=str, default='/mnt/nas_10/group/hongjia/PanopticGS-test/Replica')
    parser.add_argument("--gt_path", type=str, default='./datasets/replica/3d_sem_ins')
    args = parser.parse_args()

    evaluate_replica(args)