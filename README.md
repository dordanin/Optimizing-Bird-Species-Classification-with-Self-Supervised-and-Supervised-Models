# Deep Learning - Spring - 00046211
Dor Danino, Tal Polak - Spring 2024


# Project documentation
## Topics
* Introduction
* Method
* Results
* Conclusions
* How to run
* Ethics Statement

## Introduction
This project aims to compare the performance of self-supervised and supervised models across different optimizers for fine-grained image classification, with a particular focus on distinguishing between similar bird species. By leveraging transfer learning, we can utilize new pre-trained models and cutting-edge optimizers to achieve high accuracy in classifying these closely related species. This project and the dataset used are both publicly available.
## Motivation
We undertook this project to explore which model, DINOv2 (a self-supervised model) or YOLOv8 (a supervised model), performs better in fine-grained image classification, while testing which optimizers gave the best result.
Previous Works
Prior works have created a great deal of models designed for this particular dataset. In particular, Bird Detection and Species Classification: Using YOLOv5 and Deep Transfer Learning Models compared the performance of YOLOv5, VGG19, InceptionV3, and EfficientNetB3, using optimizers such as AdamW, SGD, and AdamMax.
However, works measuring the performance of newer models, such as DINOv2 and YOLOv8, do not currently exist.
# Method
In this section we’ll discuss the models, training, optimizers,  and hyperparameters.
## DINOv2 
DINOv2 (self-DIstillation with NO labels v2) is an open-source self-supervised framework that leverages Vision Transformers. Developed by researchers at Meta, it was trained on 142 million unlabeled images. We used the DINOv2 large image classification model, which contains 300 million parameters, as a feature extractor and trained the last layer as a classification layer. Using this model, we trained a total of 55 different models, each utilizing a unique combination of parameters, optimizers, or augmentations.
Each one of the models was trained with one of the following optimizers: SGD, Adam, AdamW, RMSProp, AdamN, AdamR, schedule free AdamW and, schedule free SGD. For each optimizer at least six models were trained, all training for 3 epochs.
All optimizers used their default settings, excluding the learning rate.
The following sets of parameters were used to create the models:
* Learning rate of 2.5e-4, batch size of 256, no augmentations with full precision.
* Learning rate of 2.5e-4, batch size of 128, kornia augmentations, and automatic mixed precision.
* Learning rate of 2.5e-4, batch size of 256, dropout rate of 0.3, no augmentations, and automatic mixed precision.
* Learning rate of 2.5e-4, batch size of 64, dropout rate of 0.3, kornia augmentations, and automatic mixed precision.
* Learning rate of 5e-4, batch size of 128, no augmentations, and automatic mixed precision.
* Learning rate of 5e-4, batch size of 128,  kornia augmentations, and automatic mixed precision.
  The kornia augmentations used were:
* RandomRotation of up to 45 degrees with a probability of 0.3.
* RandomHorizantalFlip with a probability of 0.3.
* RandomVerticalFlip with a probability of 0.3.
* RandomAffine of up to 30 degrees with a probability of 0.3.

Data augmentations that changed the color of the image, or erased parts of it were not used, because we believed that they might make it physically impossible for the model to accurately classify the image, as they change or remove vital information for classification.
## YOLOv8
YOLOv8 is the eighth version of YOLO (You Only Look Once), an open-source state-of-the-art model designed by Ultralytics. It is built on a CNN architecture and is tailored for various tasks, including object detection and tracking, instance segmentation, image classification, and pose estimation. The YOLOv8 image classification models were all trained on the ImageNet dataset, which contains over 14 million images across 1000 classes. We mainly used the YOLOv8n-cls model which has 2.7 million parameters, but we also tested the YOLOv8l-cls model which has 37.5 million parameters. Using transfer learning, we fine-tuned the models and created 46 different models, each using a different combination of parameters, optimizers, and augmentations.
Each one of the models was trained with one of the following optimizers: SGD, Adam, AdamW, RMSProp, AdamN, and AdamR. For each optimizer, at least 8 models were trained all training for 10 epochs. The models were trained using the Ultralytics library, using mainly the default settings, aside from those that were changed as parameters.
The following sets of parameters were used to create the models:
* Default learning rate and momentum, batch size of 32, no augmentations, and automatic mixed precision using the YOLOv8n-cls model.
* Default learning rate and momentum, batch size of 32, default augmentations, and automatic mixed precision using the YOLOv8n-cls model.
* Default learning rate and momentum, batch size of 32, manual augmentations, and automatic mixed precision using the YOLOv8n-cls model.
* Default learning rate and momentum, batch size of 32, no augmentations, dropout of 0.3, and automatic mixed precision using the YOLOv8n-cls model.
* Default learning rate and momentum, batch size of 32, default augmentations, dropout of 0.3, and automatic mixed precision using the YOLOv8n-cls model.
* Default learning rate and momentum, batch size of 32, no augmentations, and automatic mixed precision using the YOLOv8l-cls model.
* Default learning rate and momentum, batch size of 32, default augmentations, and automatic mixed precision using the YOLOv8l-cls model.
* Default learning rate and momentum, batch size of 32, manual augmentations, and automatic mixed precision using the YOLOv8l-cls model.

The default set of augmentations includes: 
* hsv_h=0.015, which controls the amount of hue adjustment in HSV (Hue, Saturation, Value) color space, which controls the saturation change in the image.
* hsv_s=0.7, controls the saturation change in the image
* hsv_v=0.4, adjusts the brightness of the image.
* translate=0.1, randomly translates the image horizontally and/or vertically by a fraction of the image size.
* scale=0.5, randomly zooms in or out.
* fliplr=0.5, flip image horizontally.
* erasing=0.4, randomly erases patches of the image.
  
The manual set of augmentations includes:
* hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5, and fliplr=0.5 from the default augmentations list
* degrees=45, rotates the image by a random degree.
* perspective=0.3, alters the image as if viewed from a different angle.


