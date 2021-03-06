import numpy as np
import tensorflow as tf
from PIL import Image
import glob
import os
# import tensorflow.contrib.slim as slim
import sys
from tensorflow import keras

import anchors as Anchors
import backbone as Backbone

############################################################################

############################################################################

def get_upsampleAndSum(to_upsample,to_add,num_filt=256):
    """
        a= upsample(to_upsample)
        b = (1x1 Conv)(to_add)
        return add[a,b] 
    """
    C4d2 = keras.layers.Conv2D(num_filt,(1,1),strides=(1,1),padding='same')(to_add)
    upsampled_shape = keras.backend.shape(C4d2)
    c = tf.image.resize(to_upsample,(upsampled_shape[1],upsampled_shape[2]),method='nearest')
    Pf = keras.layers.add([C4d2,c])
    return Pf

def get_fpn_featureMaps(features,num_filt=256):
    """
    features: list of different maps produced by a backbone
    **Resnet50** : for resnet50 it contains all the 5 feature maps from each block
    Return: list of pyramid features
    """
    C1,C2,C3,C4,C5 = features
    
    C5d1 = keras.layers.Conv2D(num_filt,(1,1),strides=(1,1),padding='same')(C5)
    P5 = keras.layers.Conv2D(num_filt,(3,3),strides=(1,1),padding='same')(C5d1)
    
    # P4d = get_upsampleAndSum(C5d1,C4,num_filt)
    P4d = get_upsampleAndSum(P5,C4,num_filt)
    P4 = keras.layers.Conv2D(num_filt,(3,3),strides=(1,1),padding='same')(P4d)
    
    # P3d = get_upsampleAndSum(P4d,C3,num_filt)
    P3d = get_upsampleAndSum(P4,C3,num_filt)
    P3 = keras.layers.Conv2D(num_filt,(3,3),strides=(1,1),padding='same')(P3d)

    ## retina face using shallower feature map
    # P2d = get_upsampleAndSum(P3d,C2,num_filt)
    P2d = get_upsampleAndSum(P3,C2,num_filt)
    P2 = keras.layers.Conv2D(num_filt,(3,3),strides=(1,1),padding='same')(P2d)
    
    P6 = keras.layers.Conv2D(num_filt,(3,3),strides=(2,2),padding='same')(C5)
    # P7d = keras.layers.ReLU()(P6)
    # P7 = keras.layers.Conv2D(num_filt,(3,3),strides=(2,2),padding='same')(P7d)
    
    # return P3,P4,P5,P6,P7
    return P2,P3,P4,P5,P6

############################################################################

############################################################################

def context_module_evaluator(name,num_filt,num_anchors,num_outputs_per_anchors=5,num_feature_filt=256,classification=False,fovial=False):
    """
    features = previous layers from the model
    num_filt = intermediate number of filters for conv
    num_anchors = num_of_anchors for this feature
    num_outputs_per_anchors = 4 for regression and 1 for classification
    return the [N,5] N is the number of anchors
    """
    tf_place = keras.layers.Input(shape=(None,None,num_feature_filt))


    ### 3x3
    x1_128 = keras.layers.Conv2D(filters=128,activation='relu'
                                      ,kernel_size=3,strides=1,padding='same')(tf_place)

    ### 5x5 1
    x2_64_1 = keras.layers.Conv2D(filters=64,activation='relu'
                                      ,kernel_size=3,strides=1,padding='same')(tf_place)

    x2_64_2 = keras.layers.Conv2D(filters=64,activation='relu'
                                      ,kernel_size=3,strides=1,padding='same')(x2_64_1)

    ### 7x7 1
    x3_64_1 = keras.layers.Conv2D(filters=64,activation='relu'
                                      ,kernel_size=3,strides=1,padding='same')(x2_64_1)

    x3_64_2 = keras.layers.Conv2D(filters=64,activation='relu'
                                      ,kernel_size=3,strides=1,padding='same')(x3_64_1)                                   

    # print('x3_64 shape',x3_64.shape)
    

    x4 = keras.layers.concatenate([x1_128,x2_64_2,x3_64_2])
    # print('x4 shape',x4.shape)


    outputs_cls = keras.layers.Conv2D(filters=num_anchors*1
                                  ,padding='same',kernel_size=1,strides=1)(x4)

    outputs_reg = keras.layers.Conv2D(filters=num_anchors*4
                                  ,padding='same',kernel_size=1,strides=1)(x4)

    if fovial:
        outputs_fovial = keras.layers.Conv2D(filters=num_anchors*10
                                  ,padding='same', kernel_size=1,strides=1)(x4)
        outputs_fovial = keras.layers.Reshape((-1,10))(outputs_fovial)

    outputs_cls = keras.layers.Reshape((-1,1))(outputs_cls)
    outputs_reg = keras.layers.Reshape((-1,4))(outputs_reg)
    
    outputs_cls = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs_cls)

    if fovial:
        model = keras.models.Model(inputs=tf_place,outputs=[outputs_cls,outputs_reg,outputs_fovial],name=name)
    else:
        model = keras.models.Model(inputs=tf_place,outputs=[outputs_cls,outputs_reg],name=name)    
    return model
                                                                     
    
def get_anchors_for_fpn(anchors_cfg,fpn_features):
    """
    Arg:
        anchors_cfg : anchors configuration for different feature maps ( dict with anchor configuration example: anchors_cfg[0]={'stride':1,base...} for fpn_features 0) 
        fpn_features : pyramid features from retinanet
    Return:
        dict of all anchors for different features 
    """

    ## get the anchors for different size feature maps 
    ## first get the reference anchors
    ref_anchors = {}
    
    for key in anchors_cfg.keys():
        ref_anchors[key] = Anchors.generate_reference_anchors(base_size=anchors_cfg[key]['base_size'],
                                                     ratios=anchors_cfg[key]['ratios'],
                                                     scales=anchors_cfg[key]['scales'])
    
    ## get the anchors for different feature maps
    anchors = {}
    
    ##fpn_features[key].shape[1].value for tf version 1.14

    for key in anchors_cfg.keys():
        print("fpn"+str(key)+": ("+str(fpn_features[key].shape[1])+","+str(fpn_features[key].shape[2])+")")
        anchors[key] = Anchors.generate_anchors_over_feature_map(fpn_features[key].shape[1],
                                                                 fpn_features[key].shape[2],
                                                                 ref_anchors=ref_anchors[key],
                                                                 stride=anchors_cfg[key]['stride']).reshape(-1,4)

    return anchors

def apply_model(model,feature):
    """
    Just an auxiliary function, will remove later
    """
    return model(feature)


############################################################################

############################################################################
#############################################################################################################################

#############################################################################################################################

def retinanet_context(input_,anchors_cfg,features,fovial=False):
    """
    Arg:
        input_ : keras input layer for taking input
        anchors_cfg : anchors configurations for different scales
        features : backbone features
    Return:
        keras Model for retinanet whose output is [N,5] first four are bbox values and last one is label
    """
    ## extract the pyramid features from the backbone features
    fpn_features = get_fpn_featureMaps(features,num_filt=256)
    
    
    names = {}
    for key in anchors_cfg.keys():
        names[key] = str(key)
    
    evaluators = {}
    for key in anchors_cfg.keys():
        evaluators[key] = context_module_evaluator(names[key]+"r",num_filt=256,num_anchors=len(anchors_cfg[key]['scales'])*len(anchors_cfg[key]['ratios']),
                                                    num_feature_filt=256,fovial=fovial)

    fpn_regression = []
    fpn_classification = []
    fpn_fovial = []
    for key in sorted(evaluators.keys()):
        if fovial:
            fpn_cls , fpn_reg, fpn_fov = apply_model(evaluators[key],(fpn_features[key]))
            fpn_fovial.append(fpn_fov)
        else:
            fpn_cls, fpn_reg = apply_model(evaluators[key],(fpn_features[key]))
        fpn_regression.append(fpn_reg)
        fpn_classification.append(fpn_cls)
        
    
    ans_regression = keras.layers.Concatenate(axis=1,name='out_regression')(fpn_regression)
    print("ans_regression_shape: ",ans_regression.shape)
    ans_classification = keras.layers.Concatenate(axis=1,name='out_classification')(fpn_classification)
    print("ans_class shape: ",ans_classification.shape)
    if fovial:
        ans_fovial = keras.layers.Concatenate(axis=1,name='out_fovial')(fpn_fovial)
        print("ans_fpn shape: ",ans_fovial.shape)
        ans = keras.layers.Concatenate(axis=2,name="out")([ans_regression,ans_fovial,ans_classification])
        print("ans shape: ",ans.shape)
    else:
        ans = keras.layers.Concatenate(axis=2,name="out")([ans_regression,ans_classification])
        print("ans shape: ",ans.shape)
    return keras.models.Model(inputs=input_,outputs=ans,name='retinanet')

############################################################################

############################################################################

def resnet50_retinanet(input_shape,anchors_cfg,fovial=False):
    """
    creates a retinanet model with input_shape and anchors_cfg with resnet50 as backbone
    returns [N,5] where N is the total number of anchors 
    """
    ## get the backbone features of resnet
    inputs = keras.layers.Input(shape=input_shape)
    model_name = 'resnet50'
    layer_index = [4,38,80,142,174]
    # layer_index = [3,37,79,141,173]
    features = Backbone.get_feature_extracting_model(input_tensor=inputs,
                                                     input_shape=input_shape,
                                                     model_name=model_name,
                                                     layer_index=layer_index)
    
    retinanet_v = retinanet_context(inputs,anchors_cfg,features,fovial)
    
    
    return retinanet_v

############################################################################
#                   POST Processing
############################################################################

def apply_regression(anchors,regression):

    anchors += regression
    return anchors
def clip_bbox(bbox,width,height):
    x1,y1,x2,y2 = tf.unstack(bbox,axis=-1)
    x1 = tf.clip_by_value(x1, 0, width)
    y1 = tf.clip_by_value(y1, 0, height)
    x2 = tf.clip_by_value(x2, 0, width)
    y2 = tf.clip_by_value(y2, 0, height)
    bbox = tf.stack([x1,y1,x2,y2],axis=2)
    return bbox
def filter_detections(ans,bbox,score_threshold=0.5,max_detections=300,nms_threshold=0.05):
    scores = ans[0]
    boxes = bbox[0]
    print(scores.shape)
    print(boxes.shape)
    indices = tf.where(tf.greater(scores,score_threshold))
    
    filtered_b = tf.gather_nd(boxes,indices)
    filtered_s = tf.gather(scores,indices)[:,0]
    
    nms_indices = tf.image.non_max_suppression(filtered_b,filtered_s,max_output_size=max_detections,iou_threshold=nms_threshold)

    nms_indices = tf.gather(indices,nms_indices)
    return nms_indices

def resnet50_retinanet_bbox(input_shape,anchors_cfg,image_shape=(480,640),fovial=False,score_threshold=0.05,nms_threshold=0.5):
    """
    creates a retinanet model with input_shape and anchors_cfg with resnet50 as backbone
    returns [N,5] where N is the total number of anchors 
    """
    ## get the backbone features of resnet
    inputs = keras.layers.Input(shape=input_shape)
    model_name = 'resnet50'
    layer_index = [4,38,80,142,174]
    # layer_index = [3,37,79,141,173]
    features = Backbone.get_feature_extracting_model(input_tensor=inputs,
                                                     input_shape=input_shape,
                                                     model_name=model_name,
                                                     layer_index=layer_index)
    
    ## get the retina_net 
    retinanet_v = retinanet_context(inputs,anchors_cfg,features,fovial)
    ## Post Processing
    anchors = Anchors.generate_anchors_from_input_shape((image_shape[0],image_shape[1]),anchors_cfg)

    anchors = tf.convert_to_tensor(anchors,dtype='float32')
    print(retinanet_v.outputs)
    anchors = tf.expand_dims(anchors,axis=0)
    print(anchors.shape)

    ## taking care of width and height
    # bbox = anchors
    anchor_width = anchors[:,:,2] - anchors[:,:,0]
    anchor_height = anchors[:,:,3] - anchors[:,:,1]
    bx1 = anchors[:,:,0]
    by1 = anchors[:,:,1]
    bx2 = anchors[:,:,2]
    by2 = anchors[:,:,3]
    bx1 += retinanet_v.outputs[0][:,:,0]*0.2*anchor_width
    by1 += retinanet_v.outputs[0][:,:,1]*0.2*anchor_height
    bx2 += retinanet_v.outputs[0][:,:,2]*0.2*anchor_width
    by2 += retinanet_v.outputs[0][:,:,3]*0.2*anchor_height
    bbox = tf.stack([bx1,by1,bx2,by2],axis=2)
    print('bx1 shape: ',bx1.shape)
    print('bbox shape now :',bbox.shape)
    bbox = clip_bbox(bbox,image_shape[0],image_shape[1])

    scores = retinanet_v.outputs[0][:,:,4]
    indices = filter_detections(scores,bbox,score_threshold=score_threshold,nms_threshold=nms_threshold)
    
    print(scores.shape)
    indices = tf.expand_dims(indices,axis=0)
    
    return keras.models.Model(inputs=inputs, outputs=[bbox,scores,indices], name="retinanet_bbox")

############################################################################

############################################################################  

def test():
    """TODO udpate the test function"""
    """
        Note: anchors_cfg is a list of dictionary with keys = ['base_size','ratios','scales','stride'] 
              for each feature map on which detection is being made 
    """
    print("Testing")
    anchors_cfg_1 = {'base_size':16,'ratios':[1],'scales':2 ** np.arange(0, 1),'stride':1}
    anchors_cfg = []
    for i in range(5):
        anchors_cfg.append(anchors_cfg_1)
    input_shape = (480,640,3)
    final_test = resnet50_retinanet(input_shape=input_shape,anchors_cfg=anchors_cfg)
    im = Image.open('../dataset/images/2019-06-13 16_25_53__000497__91b35108-8da2-11e9-827b-cf5f9d5a17ad.jpg')
    im1 = np.array(im,dtype=np.float32)
    im1 = np.expand_dims(im1,axis=0)
    print("image shape: ",im1.shape)
    import time
    a = time.time()
    b = final_test.predict(im1)
    c = time.time()
    print('time take: ',c-a)

if __name__ == "__main__":
    print("this is model.py, use train.py for training and evaluate.py for results")