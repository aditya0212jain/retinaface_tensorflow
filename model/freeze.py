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
import cv2

sys.path.append('../')

import anchors as Anchors
import backbone as Backbone
import model as Model
import dataprocess as DataHandler
import loss as Losses
import generator as Generator

from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
import numpy as np



# Clear any previous session.
tf.keras.backend.clear_session()


save_pb_dir = './model'
model_fname = './freezed_model.h5'


from tensorflow.python.tools import freeze_graph

# def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
#     with graph.as_default():
#         graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
#         graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
#         graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
#         return graphdef_frozen



# This line must be executed before loading Keras model.
tf.keras.backend.set_learning_phase(0) 

# model = load_model(model_fname)

## REtinaFAce shallower layers
anchors_cfg = {}
anchors_cfg[0] = {'base_size':16,'ratios':[1],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':4}
anchors_cfg[1] = {'base_size':32,'ratios':[1],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':8}
anchors_cfg[2] = {'base_size':64,'ratios':[1],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':16}
anchors_cfg[3] = {'base_size':128,'ratios':[1],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':32}
anchors_cfg[4] = {'base_size':256,'ratios':[1],'scales':np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),'stride':64}

input_shape = (480,640,3)#(480,640,3)


# model = Model.resnet50_retinanet(input_shape=input_shape,anchors_cfg=anchors_cfg)
model = Model.resnet50_retinanet_bbox(input_shape=input_shape,anchors_cfg=anchors_cfg,
                                      separate_evaluators=True,context=True,score_threshold=0.05,nms_threshold=0.5,
                                      image_shape=(640,640)
                                     )

## Update this weight
model.load_weights('./test_00000002.h5')

# # session = tf.compat.v1.Session()
# session = tf.Session()

# INPUT_NODE = [t.op.name for t in model.inputs]
# OUTPUT_NODE = [t.op.name for t in model.outputs]
# print(INPUT_NODE, OUTPUT_NODE)
# frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=save_pb_dir)


x = model.inputs[0]


model.outputs[0]

save_dir = "./tmp_{:%Y-%m-%d_%H%M%S}".format(datetime.datetime.now())
tf.saved_model.simple_save(tf.keras.backend.get_session(),
                           save_dir,
                           inputs={"input": model.inputs[0]},
                           outputs={"output": model.outputs[0]})

freeze_graph.freeze_graph(None,
                          None,
                          None,
                          None,
                          model.outputs[0].op.name,
                          None,
                          None,
                          os.path.join(save_dir, "frozen_model.pb"),
                          False,
                          "",
                          input_saved_model_dir=save_dir)

model.summary()