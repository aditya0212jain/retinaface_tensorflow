import numpy as np
import tensorflow as tf
from PIL import Image
import glob
import os
import time
import sys
sys.path.append('./cython/')
import iou

current_milli_time = lambda: int(round(time.time() * 1000))

# import tensorflow.contrib.slim as slim

############################################################################

############################################################################
def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

############################################################################

############################################################################

def generate_features_shape(image_shape,feature_levels):
    img_shape = np.array(image_shape[:2])
    feature_levels = np.array(feature_levels)+2
    feature_shapes = [(img_shape+2**x -1)//(2**x) for x in feature_levels]
    return feature_shapes


def generate_reference_anchors(base_size=16,ratios=[1,1.5],scales=2 ** np.arange(0, 3)):
    """
    Given a base size, ratios and scales it generates all the reference anchors 
    return [len(ratios)*len(scales),4] array of all anchors
    """
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
#     print(base_anchor)
    ratio_anchors = _ratio_enum(base_anchor, ratios)
#     print(ratio_anchors)
#     print(scales)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    return anchors

def generate_anchors_over_feature_map(f_h,f_w,ref_anchors,stride=1):
    """
    ref_anchors : (N,4) xmin,ymin,xmax,ymax
    Returns : (f_h,f_w,N,4) anchors over all the feature map
    """
    N = ref_anchors.shape[0]
    anchors = np.zeros((f_h,f_w,N,4),dtype=np.float64)
    for h in range(f_h):
        move_h = stride * h
        for w in range(f_w):
            move_w = stride*w
            for n in range(N):
                anchors[h,w,n,0] = ref_anchors[n,0] + move_w
                anchors[h,w,n,1] = ref_anchors[n,1] + move_h
                anchors[h,w,n,2] = ref_anchors[n,2] + move_w
                anchors[h,w,n,3] = ref_anchors[n,3] + move_h
    return anchors

def generate_anchors_from_input_shape(image_shape,anchors_cfg):
    feature_levels = sorted(anchors_cfg.keys())
    all_anchors = np.zeros((0,4))
    feature_shapes = generate_features_shape(image_shape,feature_levels)
    for i,fs in enumerate(feature_shapes):
        ref_anchors = generate_reference_anchors(anchors_cfg[feature_levels[i]]['base_size'],
                                                anchors_cfg[feature_levels[i]]['ratios'],
                                                anchors_cfg[feature_levels[i]]['scales'])
        anchors = generate_anchors_over_feature_map(fs[0],fs[1],ref_anchors,anchors_cfg[feature_levels[i]]['stride'])
        all_anchors = np.append(all_anchors,anchors.reshape(-1,4),axis=0)
    return all_anchors

############################################################################

############################################################################

def get_iou_Anchors(anchors,gt_boxes):
    """
    anchors: [N,4] all anchors for a feature map
    gt_boxes: [M,5] all gt_boxes with label
    return:
    iou array: [N,M]
    """
    N = anchors.shape[0]
    M = gt_boxes.shape[0]
    iou_a = np.zeros((N,M))
    for i in range(M):
        gt_box_area = (gt_boxes[i][2]-gt_boxes[i][0]+1)*(gt_boxes[i][3]-gt_boxes[i][1]+1)
        for n in range(N):
            intersection_box_width = min(gt_boxes[i][2],anchors[n][2]) - max(gt_boxes[i][0],anchors[n][0])+1
            if intersection_box_width>0:
                intersection_box_height = min(gt_boxes[i][3],anchors[n][3]) - max(gt_boxes[i][1],anchors[n][1])+1
                if intersection_box_height>0:
                    intersection_area = intersection_box_width*intersection_box_height
                    anchor_area = (anchors[n][2]-anchors[n][0]+1)*(anchors[n][3]-anchors[n][1]+1)
                    union_area = gt_box_area+anchor_area-intersection_area
                    iou_a[n][i] = intersection_area/union_area
    return iou_a

def get_regression_target_values(anchors,gt_boxes):
    """
    for each anchor and its corresponding max iou gt_box returns the regression values 
    anchors: [N,4]
    gt_boxes : [N,5]
    returns :[N][4] <- regression values to be predicted by the network
    N is the number of anchors 
    """
    mean = np.array([0,0,0,0])
    std = np.array([0.2,0.2,0.2,0.2])
    
    anchors_widths = anchors[:,2] - anchors[:,0]
    anchors_height = anchors[:,3] - anchors[:,1]
    
    regress_x1 = (gt_boxes[:,0] - anchors[:,0])/anchors_widths
    regress_y1 = (gt_boxes[:,1] - anchors[:,1])/anchors_height
    regress_x2 = (gt_boxes[:,2] - anchors[:,2])/anchors_widths
    regress_y2 = (gt_boxes[:,3] - anchors[:,3])/anchors_height
    
    regress_target = np.stack((regress_x1,regress_y1,regress_x2,regress_y2))
    regress_target = regress_target.T
    
    regress_target = (regress_target-mean)/std

    return regress_target

############################################################################

############################################################################

def get_regression_and_labels_values(anchors,gt_boxes,image_shape=None,positive_threshold=0.5,negative_threshold=0.4):
    """
    anchors: [N,4] all anchors for a feature map
    gt_boxes: [M,5] all gt_boxes with label
    returns :
    labels : [N,1+1] additional value for ignoring or not (1->positive 0 for negative, -1 for ignoring)
    regression : [N,4+1] additional value for ignoring or not
    """
    labels = np.zeros((anchors.shape[0],2))
    regression = np.zeros((anchors.shape[0],5))
    ## get all the ious between all anchors and all gt_boxes
    t1 = current_milli_time()
    all_iou = iou.get_iou(anchors,gt_boxes)
    # print("time in getting ious: ",current_milli_time()- t1)
    ## get the gt_box index with maximum overlap for each anchor
    max_overlap = np.argmax(all_iou,axis=1)
    ## get the value of that max overlap
    max_overlap_v = all_iou[np.arange(all_iou.shape[0]),max_overlap]
    
    ## finding positive ,ignored and negative anchors
    positive_anchor_indices = max_overlap_v >= positive_threshold
    ignored_anchor_indices = (max_overlap_v>negative_threshold) & ~positive_anchor_indices
    
    ## set which ones to consider
    labels[ignored_anchor_indices,-1] = -1
    labels[positive_anchor_indices,-1] = 1
    regression[ignored_anchor_indices,-1] = -1
    regression[positive_anchor_indices,-1] = 1
    
    ## setting label as 1 for positive anchors (0 for negative)
    labels[positive_anchor_indices,0] = 1
    ## setting the regression values for each anchor corresponding to its max overlapping gt_box
    regression[:,:-1] = get_regression_target_values(anchors,gt_boxes[max_overlap])
    
    if image_shape!=None:
        ## filtering anchors with center outside the image shape
        anchors_center = np.stack(((anchors[:,2] + anchors[:,0])/2,(anchors[:,3]+anchors[:,1])/2)).T
        ## getting indices of anchors outside the image: image_shape= (h,w) of the image
        outside_indices_x = np.logical_or(anchors_center[:,0]>=int(image_shape[1]),anchors_center[:,0]<0)
        outside_indices_y = np.logical_or(anchors_center[:,1]>=image_shape[0],anchors_center[:,1]<0)
        ## ignoring outside anchors
        labels[outside_indices_x,-1] = -1
        regression[outside_indices_x,-1] = -1
        labels[outside_indices_y,-1] = -1
        regression[outside_indices_y,-1] = -1
    
    return regression, labels

def get_regression_and_labels_batch(anchors,image_batch,annotations_batch,positive_threshold=0.5,negative_threshold=0.4):
    regression_batch = []
    label_batch = []

    annotations_batch = annotations_batch['bbox']

    for (image,annotation) in zip(image_batch,annotations_batch):
        # if annotation['bbox'].shape[0]:
        # print(annotation)   
        if annotation.any():
            print(image.shape)
            regression , labels = get_regression_and_labels_values(anchors,np.asarray(annotation),image.shape)
            regression_batch.append(regression)
            label_batch.append(labels)
        else:
            labels = np.zeros((anchors.shape[0],2))
            regression = np.zeros((anchors.shape[0],5))
            regression_batch.append(regression)
            label_batch.append(labels)
    
    return np.asarray(regression_batch) , np.asarray(label_batch)