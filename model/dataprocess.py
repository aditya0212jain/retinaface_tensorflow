import numpy as np
import tensorflow as tf
from PIL import Image
import glob
import os
import xml.etree.ElementTree as ET

class WiderDataset:
    """
    [left, top, width, height, score]
    """
    def __init__(self,image_folder_path,label_path,style="tlwh"):
        self.style = style
        im_paths = self.get_image_paths(image_folder_path)
        image_labels, image_fovials = self.get_labels(label_path)
        self.data = self.get_image_unified(im_paths,image_labels,image_fovials)
        return 
        
    def get_image_paths(self,folder_path="../dataset/Wider/WIDER_train/images/"):
        folders = glob.glob(folder_path+"*")
        images_paths = []
        for folder in folders:
            images_paths.extend(glob.glob(os.path.join(folder,'*')))
        return images_paths
    
    def get_labels(self,label_file_path):
        label_file = open(label_file_path,'r')
        ## Getting image names
        image_names = []
        flag = False
        image_labels = {}
        image_fovials = {}
        for line in label_file:
            sp = line.strip().split('/')
            if len(sp)>1:
                image_names.append(sp[-1])
                image_labels[image_names[-1]] = []
                image_fovials[image_names[-1]] = []
                continue
            cor = line.strip().split(' ')
            if len(cor)==1:
                continue
            image_labels[image_names[-1]].append([int(val) for val in cor[:4]])
            if len(cor)>18:
                image_fovials[image_names[-1]].append([float(val) for val in cor[4:19]])
        label_file.close()
        return image_labels, image_fovials
        
    def get_image_unified(self,image_paths,image_lables_dict,image_fovials_dict):
        data = []
        for im_p in image_paths:
            im_name = im_p.strip().split('/')[-1]
            data_object = {}
            data_object['path'] = im_p
            data_object['bbox'] = image_lables_dict[im_name]
            data_object['fovial'] = image_fovials_dict[im_name]
            data.append(data_object)
        return data
    
    def get_item(self,i,style='anchor'):
        ## returns image array and labels
        ## if style is anchor then returns bbox in xmin,ymin,xmax,ymax 
        ## else in xmin,ymin,width,height
        image = np.array(Image.open(self.data[i]['path']))
        label = self.data[i]['bbox']
        if style=='anchor':
            label = self.get_anchor_type(label)
        ### do the preprocessing on the image here
        ## TODO 
        # image = self.preprocess(image)
        return image,label
    
    def get_anchor_type(self,labels):
        new_label = []
        for l in labels:
            new_label.append([l[0],l[1],l[0]+l[2],l[1]+l[3]])
        return new_label
        
    
    def preprocess(self,image):
        return image

class maviData:
    """
    [left, top, width, height, score]
    each object : {'bbox': ,'path':}
    """
    def __init__(self,image_root_dir,annotations_root_dir):
        im_paths = glob.glob(image_root_dir+"*")
        image_labels = self.get_labels(annotations_root_dir)
        self.data = self.get_image_unified(im_paths,image_labels)
        return
        
    def get_labels(self,annotations_root_dir):
        ann_locs = glob.glob(annotations_root_dir+"*")
        image_labels ={}
        for ann in ann_locs:
            bboxs = self.get_topLeftWidthHeight(ann)
            im = self.get_last(ann)[:-4]
            image_labels[im] = bboxs
        return image_labels
    
    def get_image_unified(self,im_paths,image_labels):
        data = []
        for path in im_paths:
#             print(path)
            dataObject = {}
            dataObject['path'] = path
            dataObject['bbox'] = image_labels[self.get_last(path)[:-4]]
            if dataObject['bbox']:
                data.append(dataObject)
        return data
        
    def get_item(self,i,style='anchor'):
        image = np.array(Image.open(self.data[i]['path']))
        label = self.data[i]['bbox']
        if style=='anchor':
            label = self.get_anchor_type(label)
        ### do the preprocessing on the image here
        ## TODO 
        # image = self.preprocess(image)
        return image,label
    
    def get_anchor_type(self,labels):
        new_label = []
        for l in labels:
            new_label.append([l[0],l[1],l[0]+l[2],l[1]+l[3]])
        return new_label
        
    
    def preprocess(self,image):
        return image
    
    def get_topLeftWidthHeight(self,filepath):
        tree = ET.parse(filepath)
        root = tree.getroot()
        annotations = []
        for bndbox in root.findall('object/bndbox'):
            xmin = bndbox.find('xmin').text
            xmax = bndbox.find('xmax').text
            ymin = bndbox.find('ymin').text
            ymax = bndbox.find('ymax').text
            x = xmin
            y = ymin 
            w = str(int(xmax) - int(xmin))
            h = str(int(ymax) - int(ymin))
            annotations.append((int(x),int(y),int(w),int(h)))
        return annotations
    
    def get_last(self,x):
        a = x.split('/')[-1]
        return a