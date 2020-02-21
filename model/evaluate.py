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

sys.path.append('../')

import anchors as Anchors
import backbone as Backbone
import model as Model
import dataprocess as DataHandler
import loss as Losses
import generator as Generator



# from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


train_loc = "../../dataset/train_split_images/"
annotations_loc = "../../dataset/annotations/"
cropped_wider = "../../dataset/Wider_cropped/train_images/"
cropped_annotations = "../../dataset/Wider_cropped/ground_truth.txt"
wider_train = "../../dataset/Wider/WIDER_train/images/"
wider_annotation = "../../dataset/Wider/wider_face_split/wider_face_train_bbx_gt.txt"
wider_val = "../../dataset/WIDER_val/images/"
wider_val_annotation = "../../dataset/Wider/wider_face_split/wider_face_val_bbx_gt.txt"

# mData = DataHandler.WiderDataset(cropped_wider,cropped_annotations)
mData = DataHandler.WiderDataset(wider_val,wider_val_annotation)
# mData = DataHandler.maviData(train_loc,annotations_loc)


## REtinaFAce shallower layers
anchors_cfg = {}
anchors_cfg[0] = {'base_size':16,'ratios':[1],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':4}
anchors_cfg[1] = {'base_size':32,'ratios':[1],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':8}
anchors_cfg[2] = {'base_size':64,'ratios':[1],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':16}
anchors_cfg[3] = {'base_size':128,'ratios':[1],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':32}
anchors_cfg[4] = {'base_size':256,'ratios':[1],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':64}

input_shape = (640,640,3)#(480,640,3)



# model = Model.resnet50_retinanet(input_shape=input_shape,anchors_cfg=anchors_cfg,separate_evaluators=True)
model = Model.resnet50_retinanet_bbox(input_shape=input_shape,anchors_cfg=anchors_cfg,
                                      separate_evaluators=True,context=True,score_threshold=0.5,nms_threshold=0.5,
                                      image_shape=(640,640)
                                     )

# model.load_weights('./28Jan_2_00000001.h5')
# model.load_weights('./29Jan_1_00000005.h5')
model.load_weights('./8Jan_1_00000006.h5')
# model.load_weights('../notebooks/5Feb_1_00000012.h5')



batch_size = 1
train_generator = Generator.Generator(mData,anchors_cfg,batch_size=batch_size,batch_by='aspect_ratio',preprocess=True,
                                     save_annotations=True,evaluation=True,save_annotations_dir="../../validation_gt_generator/")

# anchors = Anchors.generate_anchors_from_input_shape((480,640),anchors_cfg)
# len(mData.data)

print("len ",train_generator.len)


save_dir = "../../evaluate_results/"
save_dir = "../../validation_det/"
# save_dir = "../../mavi_det/"

def save_results(save_dir,image_name,boxes,scores):
    write_path = save_dir + image_name[:-3]+"txt"
    f = open(write_path,'w')
    if len(boxes)>300:
        boxes = boxes[:300]
    for i,box in enumerate(boxes):
        s = "face " + str(scores[i])+" "
        for b in box:
            s += str(int(b)) + " "
        s += "\n"
        f.write(s)
    f.close()

num_data = len(mData.data)
# num_data = 1000

for i in range(num_data):
    t,path = train_generator.__getitem__(i)
    ans = model.predict(t)
    bbox = ans[0][0]
    scores= ans[1][0]
    indices = ans[2][0][:,0]
    bboxes = np.take(bbox,indices,axis=0)
    scores = np.take(scores,indices,axis=0)
    save_results(save_dir,path.split('/')[-1],np.array(bboxes),np.array(scores))
    print(i)







