import numpy as np
import tensorflow as tf
from PIL import Image
import glob
import os
from tensorflow import keras
import anchors as Anchors
import random

class Generator(keras.utils.Sequence):
    """
    Generator class for training
    """

    def __init__(self,dataset,anchors_cfg,batch_size=1,batch_by='random',preprocess=False,max_shape=None):
        self.len = len(dataset.data) 
        self.do_preprocess = preprocess
        self.image_paths , self.annotations = self.get_PathsAndAnnotations_dataset(dataset)
        self.batch_size = batch_size
        self.anchors_cfg = anchors_cfg
        self.make_batches()
        if max_shape is not None:
            self.anchors = Anchors.generate_anchors_from_input_shape(max_shape,self.anchors_cfg)
        else:
            self.anchors = None
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
        # print(self.image_paths[batch[0]])
        for i in range(len(batch)):
            image_batch.append(np.asarray(Image.open(self.image_paths[batch[i]]).convert('RGB')))
        return image_batch

    def get_cropped_annotations(self,boxes,image_shape):
        xoff = (image_shape[0]-480)/2
        yoff = (image_shape[1]-640)/2
        new_boxes = []
        for i,box in enumerate(boxes):
            nbox = [box[0]-xoff,box[1]-yoff,box[2]-xoff,box[3]-yoff]
            if min(nbox)>=0:
                if nbox[0]<480 and nbox[2]<480 :
                    if nbox[1]<640 and nbox[3]<640:
                        new_boxes.append(nbox)
        return new_boxes
        
    
    def preprocess_old(self,image_batch,annotations_batch):
        #TODO : add preprocessing like horizontal flip etc.
        cropped_image = []
        new_boxes = []
        for i,image in enumerate(image_batch):
            if image.shape[0]>480 and image.shape[1]>640:
                cropped_image.append(np.array(tf.image.resize_with_crop_or_pad(image,480,640)))
                new_boxes.append(self.get_cropped_annotations(annotations_batch['bbox'][i],image.shape))
            else:
                cropped_image.append(image)
                new_boxes.append(annotations_batch['bbox'][i])
        image_batch = cropped_image
        annotations_batch['bbox'] = new_boxes
        return image_batch,annotations_batch

    def modify_annotations(self,x_off,y_off,annotations,square_size,to_width=640,to_height=640):
        ### first get the annotations only in the square box
        new_boxes = []
        boxes = annotations
        for i,box in enumerate(boxes):
                nbox = [box[0]-x_off,box[1]-y_off,box[2]-x_off,box[3]-y_off]
                flag = True
                if min(nbox)>=0:
                    if nbox[0]<to_width and nbox[2]<to_height :
                        if nbox[1]<to_width and nbox[3]<to_height:
                            new_boxes.append(nbox)
                        else:
                            flag = False
                    else:
                        flag = False
                else:
                    flag= False
                if flag == False:
                    if (nbox[0]+nbox[2])/2<640:
                            if (nbox[1]+nbox[3])/2 < 480:
                                new_boxes.append(nbox)
        ## secondly resize them 
        ratio = float(to_width)/float(square_size)
        new_boxes = np.array(new_boxes)
        new_boxes = new_boxes * ratio
        return new_boxes

    def preprocess(self,image_batch,annotations_batch):
        # print(annotations_batch['bbox'])
        cropped_image = []
        new_boxes = []
        for i,image in enumerate(image_batch):
            # print(image.shape)
            o_width = image.shape[1]
            o_height = image.shape[0]
            sz = min(o_height,o_width)
            square_size = random.randint(int(0.3*sz),int(sz))
            # print(square_size)
            x_off = random.randint(0,o_width - square_size)
            y_off = random.randint(0,o_height - square_size)
            boxes = np.zeros((1,4))
            # print(boxes)
            boxes[0] = [y_off/o_height,x_off/o_width,(y_off+square_size)/o_height,(x_off+square_size)/o_width]
            # imp = tf.keras.preprocessing.image.img_to_array(im)
            imp = tf.expand_dims(image,axis=0)
            #im_cropped = tf.image.crop_and_resize(imp,boxes,np.array([0]),np.array([640,640]))
            to_width = 640
            im_cropped = tf.image.crop_and_resize(imp,boxes,np.array([0]),np.array([to_width,to_width]))
            # im_cropped = tf.image.crop_to_bounding_box(imp,x_off,y_off,square_size,square_size)
            cropped_image.append(np.array(im_cropped[0]))
            
            nb = self.modify_annotations(x_off,y_off,annotations_batch['bbox'][i],square_size,to_width,to_width)
            # cropped_image.append(np.array(tf.image.resize_with_crop_or_pad(image,480,640)))
            # new_boxes.append(self.get_cropped_annotations(annotations_batch['bbox'][i],image.shape))
            new_boxes.append(nb)
            # else:
            #     cropped_image.append(image)
            #     new_boxes.append(annotations_batch['bbox'][i])
        image_batch = cropped_image
        annotations_batch['bbox'] = new_boxes
        return image_batch,annotations_batch
    
    def get_inputs(self, image_batch):

        print(image_batch[0].shape)
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_batch) for x in range(3))
        # print(max_shape)
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
        # anchors = Anchors.generate_anchors_from_input_shape(max_shape,self.anchors_cfg)
        if self.anchors is not None:
            anchors = self.anchors
        else:
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

        if self.do_preprocess:
            image_batch , annotations_batch = self.preprocess(image_batch,annotations_batch)
        else:
            annotations_batch['bbox'] = [np.array(ann) for ann in annotations_batch['bbox']]

        inputs = self.get_inputs(image_batch)
        targets = self.get_targets(image_batch,annotations_batch)

        return inputs,targets

    def __len__(self,):
        return len(self.batches)

    def __getitem__(self,index):
        batch = self.batches[index]
        print(batch)
        inputs, targets = self.get_inputs_and_targets(batch)
        targets = np.concatenate([targets[0][:,:,:4],targets[1]],axis=2)
        return inputs,targets