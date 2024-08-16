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

Data augmentations that changed the color of the image, or erased parts of it were not used, because we believed that they might make it physically impossible for the model to accurately classify the image, as they change or remove vital information for classification.
## YOLOv8
YOLOv8 is the eighth version of YOLO (You Only Look Once), an open-source state-of-the-art model designed by Ultralytics. It is built on a CNN architecture and is tailored for various tasks, including object detection and tracking, instance segmentation, image classification, and pose estimation. The YOLOv8 image classification models were all trained on the ImageNet dataset, which contains over 14 million images across 1000 classes. We mainly used the YOLOv8n-cls model which has 2.7 million parameters, but we also tested the YOLOv8l-cls model which has 37.5 million parameters. Using transfer learning, we fine-tuned the models and created 46 different models, each using a different combination of parameters, optimizers, and augmentations.
Each one of the models was trained with one of the following optimizers: SGD, Adam, AdamW, RMSProp, AdamN, and AdamR. For each optimizer, at least 8 models were trained all training for 10 epochs. The models were trained using the Ultralytics library, using mainly the default settings, aside from those that were changed as parameters.

The sets of parameters were used to create DINMOv2 and YOLOv8 models appear in the models parameters document.txt file 



