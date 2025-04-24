# Image-Classfication-with-ResNet50
# Description
This project focuses on building an image classification model using transfer learning with ResNet50. We fine-tune a pretrained ResNet50 on a reduced version of the ImageNet dataset, which contains 1000 classes. The model is evaluated using top-1 accuracy and loss.

Dataset
We used a resized dataset from Huggingface. 
This is the link to the dataset:  [`evanarlian/imagenet_1k_resized_256`](https://huggingface.co/datasets/evanarlian/imagenet_1k_resized_256)]
The dataset is resized to 256x256, but we resized it again to 224x224 during the preprocess for ResNet50
We used only 20% of the dataset to reduce the compute cost, but if there is a stronger GPU avaible 100% of the dataset should be good. 

How to run
