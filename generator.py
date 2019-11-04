import numpy as np
import tensorflow as tf
from PIL import Image
import glob
import os
from tensorflow import keras
import anchors as Anchors

class Generator(keras.utils.Sequence):
    """
    Generator class for training
    """

    def __init__(self,dataset,anchors_cfg,batch_size=1):
        self.image_paths , self.annotations = self.get_PathsAndAnnotations_dataset(dataset)
        self.len = len(dataset.data) 
        self.batch_size = batch_size
        self.anchors_cfg = anchors_cfg
        self.make_batches()
        return

    def get_PathsAndAnnotations_dataset(self,dataset):
        paths = []
        annotations = []
        for i in range(self.len):
            paths.append(dataset.data[i]['path'])
            annotations.append(self.get_anchor_type(dataset.data[i]['bbox']))
        return paths, annotations

    def get_anchor_type(self,labels):
        new_label = []
        for l in labels:
            new_label.append([l[0],l[1],l[0]+l[2],l[1]+l[3]])
        return new_label

    def make_batches(self,by='aspect_ratio'):
        indices = list(range(self.len))

        if by=='random':
            random.shuffle(indices)
        elif by=='aspect_ratio':
            indices.sort(key=lambda x: self.get_aspect_ratio(x))

        self.batches = []
        for i in range(0,self.len,self.batch_size):
            one_batch = []
            for j in range(i,i+self.batch_size):
                one_batch.append(indices[j%(self.len)])
            self.batches.append(one_batch)
    
    def get_aspect_ratio(self,image_index):
        img = Image.open(self.image_paths[image_index])
        return float(img.width)/float(img.height)

    def load_image_batch(self,batch):
        image_batch = []
        for i in range(len(batch)):
            image_batch.append(np.asarray(Image.open(self.image_paths[batch[i]]).convert('RGB')))
        return image_batch

    def preprocess(self,image_batch,annotations_batch):
        #TODO : add preprocessing like horizontal flip etc.
        return image_batch,annotations_batch

    def get_inputs(self, image_batch):

        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_batch) for x in range(3))

        # construct an image batch object
        inputs = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_batch):
            inputs[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        # if keras.backend.image_data_format() == 'channels_first':
        #     inputs = inputs.transpose((0, 3, 1, 2))

        return inputs

    def get_targets(self,image_batch,annotations_batch):
        """
        First generate the anchors for maxsize input image, then compute the labels and regression target values 
        Generating anchors :
            this would require to get the feature map size
        """
        max_shape = tuple(max(image.shape[x] for image in image_batch) for x in range(3))
        anchors = Anchors.generate_anchors_from_input_shape(max_shape,self.anchors_cfg)

        targets = Anchors.get_regression_and_labels_batch(anchors,image_batch,annotations_batch)

        return list(targets)

    def on_epoch_end(self):
        np.random.shuffle(self.batches)
        return

    def get_inputs_and_targets(self,batch):

        image_batch = self.load_image_batch(batch)
        annotations_batch = {'label':[1 for i in range(len(batch))],'bbox':[self.annotations[i] for i in batch]}

        ## do preprocessing

        image_batch , annotations_batch = self.preprocess(image_batch,annotations_batch)

        inputs = self.get_inputs(image_batch)
        targets = self.get_targets(image_batch,annotations_batch)

        return inputs,targets

    def __len__(self,):
        return len(self.batches)

    def __getitem__(self,index):
        batch = self.batches[index]
        inputs, targets = self.get_inputs_and_targets(batch)
        return inputs,targets