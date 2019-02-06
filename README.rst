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

  pip install git+https://github.com/petko-nikolov/pysemseg
    
   
Models
======

- FCN [`paper <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_] - FCN32, FCN16, FCN8 with pre-trained VGG16
- UNet [`paper <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_]
- Tiramisu (FC DenseNets)[`paper <https://arxiv.org/pdf/1611.09326.pdf>`_] - FC DenseNet 56, FC DenseNet 67, FC DensetNet 103 with efficient checkpointing
- DeepLab V3 [`paper <https://arxiv.org/pdf/1706.05587.pdf>`_] - Multi-grid, ASPP and BatchNorm fine-tuning with pre-trained resnets backbone
- DeepLab V3+ [`paper <https://arxiv.org/pdf/1802.02611.pdf>`_]
- RefineNet [`paper <https://arxiv.org/pdf/1611.06612.pdf>`_] - [Upcoming ...]


Datasets
========
- `Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_
- `CamVid <http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/>`_
- Cityscapes [Upcoming ...]
- ADE20K [Upcoming ...]


Train a model from command line
===============================

The following is an example command to train a VGGFCN8 model on the Pascal VOC 2012 dataset. In addition to the dataset and the model, a transformer class should be passed (PascalVOCTransform in this case) - a callable where all input image and mask augmentations and tensor transforms are implemented. Run :code:`pysemseg-train -h` for a full list of options.

.. code:: bash

 pysemseg-train \
    --model VGGFCN8 \
    --model-dir ~/models/vgg8_pascal_model/ \
    --dataset PascalVOCSegmentation \
    --data-dir ~/datasets/PascalVOC/ \
    --batch-size 4 \
    --test-batch-size 1 \
    --epochs 40 \
    --lr 0.001 \
    -- optimizer SGD \
    -- optimizer-args '{"weight_decay": 0.0005, "momentum": 0.9}' \
    --transformer PascalVOCTransform \
    --lr-scheduler PolyLR \
    --lr-scheduler_args '{"max_epochs": 40, "gamma": 0.8}'
    
   
or pass a YAML config



.. code:: bash

    pysemseg-train --config config.yaml


.. code:: YAML

    model: VGGFCN32
    model-dir: models/vgg8_pascal_model/
    dataset: PascalVOCSegmentation
    data-dir: datasets/PascalVOC/
    batch-size: 4
    test-batch-size: 1
    epochs: 40
    lr: 0.001
    optimizer: SGD
    optimizer-args:
        weight_decay: 0.0005
        momentum: 0.9
    transformer: PascalVOCTransform
    no-cuda: true
    lr-scheduler: PolyLR
    lr-scheduler-args:
        max_epochs: 40
        gamma: 0.8

Load and predict with a trained model
=====================================

To use a checkpoint for inference you have to call :code:`load_model` with a checkpoint, the model class and the transformer class used during training.

.. code:: python

   imoprt torch.nn.functional as F
   from pysemseg.transforms import CV2ImageLoader
   from pysemseg.utils import load_model
   from pysemseg.models import VGGFCN32
   from pysemseg.datasets import PascalVOCTransform
   
   model = load_model(
       './checkpoint_path', 
       VGGFCN32, 
       PascalVOCTransform
   )
   
   image = CV2ImageLoader()('./image_path')
   logits = model(image)
   probabilities = F.softmax(logits, dim=1)
   predictions = torch.argmax(logits, dim=1)
