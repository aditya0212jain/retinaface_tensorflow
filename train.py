import numpy as np
import tensorflow as tf
from PIL import Image
import glob
import os
# import tensorflow.contrib.slim as slim
import sys
from tensorflow import keras
import time

sys.path.append('../')

import anchors as Anchors
import backbone as Backbone
import model as Model
import dataprocess as DataHandler
import loss as Losses
import generator as Generator

image_folder_path = "../dataset/Wider/WIDER_train/images/"
label_path = "../dataset/Wider/wider_face_split/wider_face_train_bbx_gt.txt"

widerDataset = DataHandler.WiderDataset(image_folder_path,label_path)

anchors_cfg_1 = {'base_size':16,'ratios':[1],'scales':2 ** np.arange(0, 1),'stride':2}
anchors_cfg = {}
anchors_cfg[0] = {'base_size':16,'ratios':[1,1.5],'scales':2 ** np.arange(0, 1),'stride':4}
anchors_cfg[1] = {'base_size':16,'ratios':[1,1.5],'scales':2 ** np.arange(1, 2),'stride':8}
anchors_cfg[2] = {'base_size':16,'ratios':[1,1.5],'scales':2 ** np.arange(2, 3),'stride':16}
anchors_cfg[3] = {'base_size':16,'ratios':[1,1.5],'scales':2 ** np.arange(3, 4),'stride':32}
anchors_cfg[4] = {'base_size':16,'ratios':[1,1.5],'scales':2 ** np.arange(4, 5),'stride':64}
input_shape = (None,None,3)#(480,640,3)

model = Model.resnet50_retinanet(input_shape=input_shape,anchors_cfg=anchors_cfg)

epochs = 1
batch_size = 1
lr = 0.01
model.compile(
    loss={
        'out': Losses.focal_plus_smooth()
    },
    optimizer=keras.optimizers.Adam(lr=lr, clipnorm=0.001)
)

train_generator = Generator.Generator(widerDataset,anchors_cfg,batch_size=batch_size,batch_by='aspect_ratio')

mc = keras.callbacks.ModelCheckpoint('weights{epoch:08d}.h5', 
                                     save_weights_only=True, save_freq='epoch')

trained = model.fit_generator(generator=train_generator,
                              steps_per_epoch=len(widerDataset.data)/(10*batch_size),
                              epochs=10*epochs,
                              workers=1,
                              max_queue_size=1
                             )