import numpy as np
import tensorflow as tf
from PIL import Image
import glob
import os
# import tensorflow.contrib.slim as slim
import sys
from tensorflow import keras
import time
import matplotlib.pyplot as plt
import pickle
import datetime 

sys.path.append('../')

import anchors as Anchors
import backbone as Backbone
import model as Model
import dataprocess as DataHandler
import loss as Losses
import generator as Generator

# from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

image_folder_path = "../dataset/Wider/WIDER_train/images/"
label_path = "../dataset/Wider/wider_face_split/wider_face_train_bbx_gt.txt"
cropped_wider = "../dataset/Wider_cropped/train_images/"
cropped_annotations = "../dataset/Wider_cropped/ground_truth.txt"

# widerDataset = DataHandler.WiderDataset(image_folder_path,label_path)
widerDataset = DataHandler.WiderDataset(cropped_wider,cropped_annotations)

# pixels = []
# for data in widerDataset.data:
#     img = Image.open(data["path"])
#     pixels.append(img.size[0]*img.size[1])
# p = np.array(pixels)
# inde = p<807200
# new_data = []
# for i,datum in enumerate(widerDataset.data):
#     if inde[i]==True:
#         new_data.append(widerDataset.data[i])
# print(len(new_data))
# widerDataset.data = new_data

# anchors_cfg_1 = {'base_size':16,'ratios':[1],'scales':2 ** np.arange(0, 1),'stride':2}
# anchors_cfg = {}
# anchors_cfg[0] = {'base_size':8,'ratios':[1,1.5],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0),2], keras.backend.floatx()),'stride':8}
# anchors_cfg[1] = {'base_size':16,'ratios':[1,1.5],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0),2], keras.backend.floatx()),'stride':16}
# anchors_cfg[2] = {'base_size':32,'ratios':[1,1.5],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0),2], keras.backend.floatx()),'stride':32}
# anchors_cfg[3] = {'base_size':64,'ratios':[1,1.5],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0),2], keras.backend.floatx()),'stride':64}
# anchors_cfg[4] = {'base_size':128,'ratios':[1,1.5],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0),2], keras.backend.floatx()),'stride':128}

## REtinaFAce shallower layers
anchors_cfg = {}
# anchors_cfg[0] = {'base_size':32,'ratios':[1,1.5],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':8}
# anchors_cfg[1] = {'base_size':64,'ratios':[1,1.5],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':16}
# anchors_cfg[2] = {'base_size':128,'ratios':[1,1.5],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':32}
# anchors_cfg[3] = {'base_size':256,'ratios':[1,1.5],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':64}
# anchors_cfg[4] = {'base_size':512,'ratios':[1,1.5],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':128}
anchors_cfg[0] = {'base_size':16,'ratios':[1],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':4}
anchors_cfg[1] = {'base_size':32,'ratios':[1],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':8}
anchors_cfg[2] = {'base_size':64,'ratios':[1],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':16}
anchors_cfg[3] = {'base_size':128,'ratios':[1],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':32}
anchors_cfg[4] = {'base_size':256,'ratios':[1],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':64}


## below one is standard
# anchors_cfg[0] = {'base_size':32,'ratios':[1,1.5],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':8}
# anchors_cfg[1] = {'base_size':64,'ratios':[1,1.5],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':16}
# anchors_cfg[2] = {'base_size':128,'ratios':[1,1.5],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':32}
# anchors_cfg[3] = {'base_size':256,'ratios':[1,1.5],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':64}
# anchors_cfg[4] = {'base_size':512,'ratios':[1,1.5],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':128}
# anchors_cfg[0] = {'base_size':16,'ratios':[1,1.5],'scales':2 ** np.arange(0, 3),'stride':4}
# anchors_cfg[1] = {'base_size':16,'ratios':[1,1.5],'scales':2 ** np.arange(0, 3),'stride':8}
# anchors_cfg[2] = {'base_size':16,'ratios':[1,1.5],'scales':2 ** np.arange(2, 4),'stride':16}
# anchors_cfg[3] = {'base_size':16,'ratios':[1,1.5],'scales':2 ** np.arange(4, 6),'stride':32}
# anchors_cfg[0] = {'base_size':16,'ratios':[1,1.5],'scales':2 ** np.arange(0, 1),'stride':4}
# anchors_cfg[1] = {'base_size':16,'ratios':[1,1.5],'scales':2 ** np.arange(1, 2),'stride':8}
# anchors_cfg[2] = {'base_size':16,'ratios':[1,1.5],'scales':2 ** np.arange(2, 3),'stride':16}
# anchors_cfg[3] = {'base_size':16,'ratios':[1,1.5],'scales':2 ** np.arange(3, 4),'stride':32}
# anchors_cfg[4] = {'base_size':16,'ratios':[1,1.5],'scales':2 ** np.arange(4, 5),'stride':64}
input_shape = (None,None,3)#(480,640,3)

model = Model.resnet50_retinanet(input_shape=input_shape,anchors_cfg=anchors_cfg)
# model.load_weights('./weights_cr_3scales_00000001.h5')

epochs = 1
batch_size = 4
lr = 0.0001
model.compile(
    loss={
        'out': Losses.focal_plus_smooth()
    },
    optimizer=keras.optimizers.Adam(lr=lr, clipnorm=0.001)
)

train_generator = Generator.Generator(widerDataset,anchors_cfg,batch_size=batch_size,batch_by='aspect_ratio',preprocess=False)

mc = keras.callbacks.ModelCheckpoint('wider_cropped_face_{epoch:08d}.h5', 
                                     save_weights_only=True, save_freq='epoch')

logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=len(widerDataset.data)/(batch_size),
                              epochs=epochs,
                              workers=4,
                              max_queue_size=10,
                              callbacks= [mc,tensorboard_callback]
                             )


pickle.dump(history.history,open('train_history_wider_cropped_face.pkl','wb'))

# plt.plot(history.history['loss'])
# plt.title('model loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()
