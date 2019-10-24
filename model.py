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
    c = keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(to_upsample)
    C4d2 = keras.layers.Conv2D(num_filt,(1,1),strides=(1,1),padding='same')(to_add)
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
    
    P4d = get_upsampleAndSum(C5d1,C4)
    P4 = keras.layers.Conv2D(num_filt,(3,3),strides=(1,1),padding='same')(P4d)
    
    P3d = get_upsampleAndSum(P4d,C3)
    P3 = keras.layers.Conv2D(num_filt,(3,3),strides=(1,1),padding='same')(P3d)
    
    P6 = keras.layers.Conv2D(num_filt,(3,3),strides=(2,2),padding='same')(C5)
    P7d = keras.layers.ReLU()(P6)
    P7 = keras.layers.Conv2D(num_filt,(3,3),strides=(2,2),padding='same')(P7d)
    
    return P3,P4,P5,P6,P7


############################################################################

############################################################################

def evaluator(name,num_filt,num_anchors,num_outputs_per_anchors=5,num_feature_filt=256):
    """
    features = previous layers from the model
    num_filt = intermediate number of filters for conv
    num_anchors = num_of_anchors for this feature
    num_outputs_per_anchors = 4 for regression and 1 for classification
    return the [N,5] N is the number of anchors
    """
    tf_place = keras.layers.Input(shape=(None,None,num_feature_filt))
    outputs = tf_place
    
    for i in range(4):
        outputs = keras.layers.Conv2D(filters=num_filt,activation='relu'
                                      ,kernel_size=3,strides=1,padding='same')(outputs)
        
    outputs = keras.layers.Conv2D(filters=num_anchors*num_outputs_per_anchors
                                  ,padding='same',kernel_size=3,strides=1)(outputs)
    
    output = keras.layers.Reshape((-1,num_outputs_per_anchors))(outputs)
    
    model = keras.models.Model(inputs=tf_place,outputs=output,name=name)
    return model
    
def get_anchors_for_fpn(anchors_cfg,fpn_features):
    """
    Arg:
        anchors_cfg : anchors configuration for different feature maps
        fpn_features : pyramid features from retinanet
    Return:
        list of all anchors for different features 
    """

    ## get the anchors for different size feature maps 
    ## first get the reference anchors
    ref_anchors = []
    for i in range(len(fpn_features)):
        ref_anchors.append(Anchors.generate_reference_anchors(base_size=anchors_cfg[i]['base_size'],
                                                     ratios=anchors_cfg[i]['ratios'],
                                                     scales=anchors_cfg[i]['scales']))
    ## get the anchors for different feature maps
    anchors = []
    for i in range(len(fpn_features)):
        anchors.append(Anchors.generate_anchors_over_feature_map(fpn_features[i].shape[1].value,
                                                                 fpn_features[i].shape[2].value,
                                                                 ref_anchors=ref_anchors[i],
                                                                 stride=anchors_cfg[i]['stride']).reshape(-1,4))
    return anchors

def apply_model(model,feature):
    """
    Just an auxiliary function, will remove later
    """
    return model(feature)


############################################################################

############################################################################

def retinanet(input_,anchors_cfg,features):
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
    
    anchors = get_anchors_for_fpn(anchors_cfg=anchors_cfg,fpn_features=fpn_features)
    
    names=[ str(i) for i in range(len(anchors))]
    
    evaluators = [ evaluator(names[i],num_filt=256,num_anchors=len(anchors[i])) for i in range(len(fpn_features)) ]
#     ans = keras.layers.Concatenate(axis=1)([ evaluator(names[i],fpn_features[i],len(anchors[i]),5)
#                                             for i in range(len(fpn_features)) ])
#   
    fpn_o = [apply_model(model,(fpn_features[i])) for i,model in enumerate(evaluators)]

    ans = keras.layers.Concatenate(axis=1)(fpn_o)
    return keras.models.Model(inputs=input_,outputs=ans,name='retinanet')
    

############################################################################

############################################################################

def resnet50_retinanet(input_shape,anchors_cfg):
    """
    creates a retinanet model with input_shape and anchors_cfg with resnet50 as backbone
    returns [N,5] where N is the total number of anchors 
    """
    ## get the backbone features of resnet
    inputs = keras.layers.Input(shape=input_shape)
    model_name = 'resnet50'
    layer_index = [6,38,80,142,174]
    features = Backbone.get_feature_extracting_model(input_tensor=inputs,
                                                     input_shape=input_shape,
                                                     model_name=model_name,
                                                     layer_index=layer_index)
    
    ## get the retina_net 
    retinanet_v = retinanet(inputs,anchors_cfg,features)
    
    return retinanet_v

def test():
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
    test()