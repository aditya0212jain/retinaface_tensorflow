import numpy as np
import tensorflow as tf
from PIL import Image
import glob
import os
# import tensorflow.contrib.slim as slim
import sys
from tensorflow import keras
import time
import datetime
import pickle

# sys.path.append('../')

import anchors as Anchors
import backbone as Backbone
import model as Model
import dataprocess as DataHandler
import loss as Losses
import generator as Generator

# get_ipython().magic('load_ext tensorboard')


# from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


print(tf.__version__)


# # Contents
# 1. Getting train and validation data in required format
# 2. Loading model
# 3. Setting training parameters: batch size, optimizer, lr, and other parameters
# 4. Get the loss function 
# 5. Writing the training process
# 6. Backpropagating 
# 7. Storing weights and testing 

train_loc = "../../dataset/train_split_images/"
annotations_loc = "../../dataset/annotations/"
cropped_wider = "../../dataset/Wider_cropped/train_images/"
cropped_annotations = "../../dataset/Wider_cropped/ground_truth.txt"
wider_train = "../../dataset/Wider/WIDER_train/images/"
wider_annotation = "../../dataset/Wider/wider_face_split/wider_face_train_bbx_gt.txt"



# mData = DataHandler.WiderDataset(cropped_wider,cropped_annotations)
mData = DataHandler.WiderDataset(wider_train,wider_annotation)
# mData = DataHandler.maviData(train_loc,annotations_loc)


pixels = []
for data in mData.data:
    img = Image.open(data["path"])
    pixels.append(img.size[0]*img.size[1])
p = np.array(pixels)
inde = p<807200
new_data = []
for i,datum in enumerate(mData.data):
    if inde[i]==True:
        new_data.append(mData.data[i])
print(len(new_data))
mData.data = new_data


len(mData.data)


# # Loading Model

## REtinaFAce shallower layers
anchors_cfg = {}
anchors_cfg[0] = {'base_size':16,'ratios':[1],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':4}
anchors_cfg[1] = {'base_size':32,'ratios':[1],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':8}
anchors_cfg[2] = {'base_size':64,'ratios':[1],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':16}
anchors_cfg[3] = {'base_size':128,'ratios':[1],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':32}
anchors_cfg[4] = {'base_size':256,'ratios':[1],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':64}

input_shape = (None,None,3)#(480,640,3)


model = Model.resnet50_retinanet(input_shape=input_shape,anchors_cfg=anchors_cfg,separate_evaluators=True)#,separate_evaluators=True


# model.load_weights('./Dec29_6_00000005.h5')
# model.load_weights('./1Jan_sep_00000008.h5')
model.load_weights('./28Jan_2_00000001.h5')


epochs = 7
batch_size = 2
# lr = 0.00003125
lr = 0.0000625#125
# 'out_classification': Losses.focal()
model.compile(
    loss={
        'out': Losses.focal_plus_smooth()
    },
    optimizer=keras.optimizers.Adam(lr=lr, clipnorm=0.001)
)


# # Preparing Generator

train_generator = Generator.Generator(mData,anchors_cfg,batch_size=batch_size,batch_by='aspect_ratio',preprocess=True)


mc = keras.callbacks.ModelCheckpoint('29Jan_1_{epoch:08d}.h5', 
                                     save_weights_only=True, save_freq='epoch')


logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


model.summary()



history1 = model.fit_generator(generator=train_generator,
                              steps_per_epoch=len(mData.data)/(batch_size),
                              epochs=epochs,
                              max_queue_size=2,
                              callbacks= [mc,tensorboard_callback]
                             )

pickle.dump(history1,open(filename+".pkl",'wb'))
