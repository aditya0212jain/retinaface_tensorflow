# TF Detector for FACEs


## Introduction

Tensorflow implementation of [Retina-Face](https://github.com/deepinsight/insightface/tree/master/RetinaFace) for porting to Movidius 2 with the help of Openvino.
Trained on [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) with Resnet50 backbone. The datapipeline is written from scratch and is easily customizable (look in model folder for changing anchor sizes or other details)

This repo contains all the scripts of model and training and evaluation. Some help was taken from
[keras-retinanet](https://github.com/fizyr/keras-retinanet/tree/master/keras_retinanet) while writing the code. 

## Installation Requirements

1) Tensorflow 1.14 (since tf 2 is not yet supported by openvino)
2) Cython
3) python 3
4) opencv

## How to Train

First compile the cython code using :

```
python setup.py build_ext --inplace
```

Then run the train code with WIDER data in ../dataset folder (For more path based queries look in the model folder)
Update the parameters as desired in the train.py script by checking the model folder, finally run:

```
python train.py 
```

## How to Evaluate

Set the detection and groud label folder in the evaluate.py script (look in model folder for more details)
Then run:

```
python evaluate.py
```

To calculate mAP we used this [repo](https://github.com/rafaelpadilla/Object-Detection-Metrics). Clone the repo and run the following command from it:

```
python pascalvoc.py -t 0.5 -gt path_to_gt_folder -det path_to_det_folder -gtformat xyrb -detformat xyrb
```

where -t is the threshold value of iou for being positive

## Creating Inference Graph for Openvino

First need to freeze the tensorflow model. After setting correct parameters run:

```
python freeze.py
```

Use the frozen graph to make inference graph using the Intel Openvino Model Optimizer. Learn more [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html) about converting a tensorflow model.

## Structure of Repo

+ <b>model</b> : contais the scripts for the model, also contains the train and evaluate scripts
+ <b>notebooks</b> : contains jupyter notebooks for experimentation and visualization

## Performance

This implementation achieved 43.95 mAP on WIDER VALIDATION (3226 images) after 9 epochs of training on the WIDER TRAINING dataset. The detector runs at approx 12 fps on Nvidia 1080.  This implementation did not use the fovial features since the aim was to make it lightweight (therefore deconv layers are also not used in the context module).

### Training Specifications

SGD optimizer and OHEM loss with smooth l1 loss for box and fovial targets

## TODO

+ Get better face detector after comparing to other detectors. 

+ Upload training graphs. 

+ Clean notebooks folder by removing unnecessary files

## References

```
@article{deng2019retinaface,
  title={Retinaface: Single-stage dense face localisation in the wild},
  author={Deng, Jiankang and Guo, Jia and Zhou, Yuxiang and Yu, Jinke and Kotsia, Irene and Zafeiriou, Stefanos},
  journal={arXiv preprint arXiv:1905.00641},
  year={2019}
}
```
