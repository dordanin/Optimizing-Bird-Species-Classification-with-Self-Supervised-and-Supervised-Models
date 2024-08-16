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

# Results
## DINOv2 Results
DINOv2 learning rate =  2.5e-4  
* Schedule-free AdamW: Achieved >99% accuracy, outperforming all other optimizers.
* SGD: Requires higher learning rates for improved accuracy. For example:
  * Without augmentation: 1e-4 → 2.32% accuracy
  * With augmentation: 2.5e-4 → 20.91%
  * Schedule-free SGD: Best results with learning rates of 1e-3 and augmentation (73.49%).

![image](https://github.com/user-attachments/assets/cd73df2c-73a1-4100-8d57-32ab54551f12)

  
DINOv2 learning rate =  2.5e-4 with dropout = 0.3 
 * Dropout: Slight positive effect on accuracy; results consistent with first figure
   
![image](https://github.com/user-attachments/assets/99a92d59-0f38-41c9-aed8-c7b2d30f951f)

   
DINOv2 learning rate =  5e-4 
 * Increased Learning Rate: Higher learning rates (5e-4) reduced accuracy
   
![image](https://github.com/user-attachments/assets/327b74f1-b71a-4628-9711-eddeeb9a7ed3)

Summary:
 * Best Performance: Schedule-free AdamW with >99% validation accuracy.
 * Effectiveness of AMP and Augmentations: Limited impact.
 * Overall Performance: Most models achieved 95%+ validation accuracy

## YOLOv8  Results
* SGD Optimizer: Best performance across all parameter sets on this dataset.
* YOLOv8l-cls vs YOLOv8n-cls did not outperform except when using SGD.
* RMSProp:
  * Performed poorly, with a maximum validation accuracy of only 43.6%.
  * Most models failed to exceed 1% accuracy, even with 12 additional attempts.
  * Likely unsuitable for YOLO or a possible implementation issue. 
* Augmentations:		
  * Models without augmentations performed better than those with.
  * Default augmentations in YOLO reduced accuracy due to image erasure.
    
![image](https://github.com/user-attachments/assets/0aa87eb2-9c4e-464b-9f54-91cb9b62091d)
![image](https://github.com/user-attachments/assets/395f5113-4ef3-4103-b6fc-4126cb1fa05c)
![image](https://github.com/user-attachments/assets/313323f2-5b69-45cb-ab4e-06ae0727cf43)


# Conclusions
DINOv2, a self-supervised learning model, achieved outstanding results, with the Schedule-free AdamW optimizer leading to a final accuracy exceeding 99%. This highlights the potential of self-supervised models in tasks requiring intricate feature extraction. The YOLOv8 model, while traditionally optimized for object detection, was fine-tuned for classification in this project. The model performed well using SGD as the optimizer, though it did not reach the accuracy levels achieved by DINOv2.

Our findings demonstrated the significant impact of augmentation techniques on the YOLOv8 model, where custom augmentations outperformed default settings. However, augmentation did not influence the performance of DINOv2 as strongly, likely due to its robust pre-training process. We also found that certain augmentations, such as erasing, could adversely affect the model by removing critical features from the images, especially for tasks requiring fine-grained detail.

In future work, further exploration into the performance of additional optimizers for YOLOv8 could provide more insights. Additionally, the inclusion of other self-supervised learning models, or hybrid methods combining both self-supervised and supervised learning techniques, could yield even more promising results for fine-grained classification.

Overall, this study underscores the strengths of both DINOv2 and YOLOv8, while also highlighting the importance of tailoring training strategies to specific model architectures and tasks.

# How to run
1) Download the data set: https://www.kaggle.com/datasets/gpiosenka/100-bird-species/data
2) Download the python files from this page
3) Add the location of the train , test and val folders that you download in 1 to train_set , val_set and test_set parameters in the code
4) Run the the python files , this will train all the models combinations

To use the best weights we found, you can download the weights file on this page and use it according the instructions:

For using YOLOv8 model, run:

`!pip install ultralytics`

`from ultralytics import YOLO`

`model = YOLO("yolov8n-cls.pt")  # load an official model`

`model = YOLO("path/to/best.pt")  # load our custom model`

`results = model("your_image.jpg")  # predict on an image`

 Using DINOv2 model : use the python file "" provide in this page 

# Ethics Statement
The explanations of ChatGPT:

stakeholders that will be affected by the project:
* Data Scientists/Researchers - who are developing and testing models for fine-grained image classification.
* Conservationists/Bird Enthusiasts - who are interested in species identification for conservation or ecological research.
* Technology Companies - that can integrate this classification technology into applications such as wildlife monitoring or mobile applications.
  
The explanation that is given to each stakeholder:
* Data Scientists/Researchers: This project explores the effectiveness of self-supervised models like DINOv2 and supervised models like YOLOv8 for bird species classification, offering insights into the performance of various optimizers. These findings can help guide the selection of model architectures and optimizers for future classification tasks.
* Conservationists/Bird Enthusiasts: This technology allows accurate identification of bird species, even those that look similar, by training AI models on a large dataset of bird images. It could assist in ecological research and help track bird populations for conservation efforts.
* Technology Companies: The models developed in this project could be used to improve wildlife tracking and monitoring systems or be integrated into consumer-facing applications like mobile apps for bird watching, offering accurate, real-time species identification.

ChatGPT explanations should emphasize the ethical considerations of AI use, particularly the potential biases in training data. For example, when explaining the project to stakeholders, it’s important to note that the dataset might not cover all species equally, potentially resulting in biased or incomplete identification results. It’s also crucial to ensure that the technology is used responsibly, especially in conservation and wildlife monitoring, to avoid harm to ecosystems or misinterpretation of results. Transparency around limitations should be included to ensure ethical use.


