# -*- coding: utf-8 -*-
"""FasterRCNN_objectDetec.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17fSR6GscZt7H35_PT8I8hss7zS8wv6QV

#Download and install Detectron2
"""

!pip install pyyaml==5.1

import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# Install detectron2 that matches the above pytorch version
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/$CUDA_VERSION/torch$TORCH_VERSION/index.html
# If there is not yet a detectron2 release that matches the given torch + CUDA version, you need to install a different pytorch.

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import PascalVOCDetectionEvaluator

"""#Download the Pascal VOC dataset"""

!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
!tar -xvf VOCtrainval_06-Nov-2007.tar

"""#Change the name of the directory """

!mv VOCdevkit datasets

"""#Train the model(fine-tune)
Here we fine tune the faste RCNN_R_50_FPN_3x model on Pascal VOC DATASET. We initialize the pre-trained weights with model_zoo.get_checkpoint_url. We set the iterations to be 3000 and batch size=128.
There are 2501 images spread acroos 20 class categories.
"""

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.OUTPUT_DIR = 'MyVOCTraining'
cfg.DATASETS.TRAIN = ("voc_2007_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml") # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025 # pick a good LR
cfg.SOLVER.MAX_ITER = 3000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

"""#Display the Quantitative results using TensorBoard"""

# Commented out IPython magic to ensure Python compatibility.
# Look at training curves in tensorboard:
# %reload_ext tensorboard
# %tensorboard --logdir MyVOCTraining/

"""#Evaluation
We load our trained model and define our predictor function (something like a forward pass with torch_no_grad). I have kept the the threshold value to be 0.7 for displaying prediction with confidence greater than 70%.
"""

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

"""#Evaluation 
We forward pass the test dataset through our trained model and print the inference report. We achieve Average Precision50 score of 67.16159. 
"""

from detectron2.evaluation import PascalVOCDetectionEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = PascalVOCDetectionEvaluator("voc_2007_val")
val_loader = build_detection_test_loader(cfg, "voc_2007_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))

"""#We now display few images and predictions of objects in each image as per our trained model. """

dataset_dicts = detectron2.data.get_detection_dataset_dicts('voc_2007_val')
for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   detectron2.data.MetadataCatalog.get('voc_2007_val'), 
                   scale=0.5, 
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])