import numpy as np
import tensorflow as tf
from PIL import Image
import glob
import os
# import tensorflow.contrib.slim as slim
import tensorflow.keras as keras

def get_feature_extracting_model(input_tensor=None,input_shape=(480,640,3),model_name='resnet50',layer_index=[6,38,80,142,174]):
    """
    input_shape : the input size of the image 
    model_name : which backbone model to be used for feature extraction
    layer_names : the names of the layer from which the outputs are to be returned
    Return: keras model with outputs of the given layers for the given model
    **Note** : Currently only works for resnet, and layer_names provided should be valid, for resnet50 the 
               results from the last layer of each block are returned
    """
    if model_name=='resnet50':
        model_i = keras.applications.ResNet50(include_top=False,weights='imagenet',input_tensor=input_tensor,input_shape=input_shape,pooling=None)
    else:
        print("Currently only support for resnet50")
        return
    C = []
    for i in layer_index:
        C.append(model_i.get_layer(model_i.layers[i].name).output)
    model = keras.models.Model(inputs = model_i.input,outputs=C)
    return C