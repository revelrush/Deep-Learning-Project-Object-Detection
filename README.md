# Drinks Object Detection Project

## About the Project
This repository contains the codebase for my assignment 2 in the deep learning elective under Dr. Atienza. In this project I implemented and trained an objected detection model using the Pytorch and the COCO library with the dataset provided by the instructor.
The finished model was capable of identifying the types of drink from different orientations and lighting conditions, the model validation accuracy was found to be ~96%. Finally, there is also a basic demo application capable of identifying a specific drink from real-time video from a webcam.

## About the Object Detection Model:
The model is high resolution Faster R-CNN model with a MobileNetV3-Large FPN backbone, this model was chosen because it's relatively lightweight compared to the other available models without compromising much performance, it's lightweightedness also made it easier to run for real-time object detection, which was critical for the video demo.

## Notes and Assumptions:
The assumption is that libraries that were used in the pytorch tutorial are all already installed (cython, pycocotools, torch, torchvision, etc.), and other libraries used to download the dataset (gdown) will be installed by train.py and test.py when they are run. In case any of the modules are missing, the command pip install -r requirements.txt can be used to install the missing modules. 

## Usage:
### Dependencies
To make sure that none of the required modules are missing, the run the command pip install -r requirements.txt on your Python Environment before running the codebase. 

### Training Script
Running the train.py file will download the dataset and run the training algorithm from scratch. Once finished, it will produce a file named model_weights.pth that can be used for inference, all you have to is rename the checkpoint file used in the demo script. 

### Demo Script
All you need to do is to run the demo.py file, it should open a simple video feed that shows which drink the model detects like this:
![image](https://github.com/revelrush/Deep-Learning-Project-Object-Detection/assets/84671795/d3f4f828-d22d-4c0b-8063-cfb1ad6ea305)

Here is a video demo that demonstrates the model in action.

https://github.com/revelrush/Deep-Learning-Project-Object-Detection/assets/84671795/93ad5d47-018b-44ee-a8d7-c7316793072d



