PySemSeg
========
 
PySemSeg is a library for training Deep Learning Models for Semantic Segmentation in Pytorch. 
The goal of the library is to provide implementations of SOTA segmentation models, with pretrained versions
on popular datasets, as well as an easy-to-use training loop for new models and datasets. Most Semantic Segmentation datasets
with fine-grained annotations are small, so Transfer Learning is crucial for success and is a core capability of the library. PySemSeg can use visdom or tensorboardX for training summary visualialization.
 
 
Installation
=============
 
Using pip:
 
.. code:: bash

  pip  git+https://github.com/petko-nikolov/pysemseg
    
   
Models
======

- FCN [`paper <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_] - FCN32, FCN16, FCN8 with pre-trained VGG16
- UNet [`paper <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_]
- Tiramisu (FC DenseNets)[`paper <https://arxiv.org/pdf/1611.09326.pdf>`_] - FC DenseNet 56, FC DenseNet 67, FC DensetNet 103
- DeepLab V3 [`paper <https://arxiv.org/pdf/1706.05587.pdf>`_] - Multi-grid, ASPP and BatchNorm fine-tuning with pre-trained resnets backbone
- DeelLab V3+ [`paper <https://arxiv.org/pdf/1802.02611.pdf>`_] - [Upcoming ...]
- RefineNet [`paper <https://arxiv.org/pdf/1611.06612.pdf>`_] - [Upcoming ...]


Datasets
========
- `Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_
- `CamVid <http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/>`_
