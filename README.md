

# Capsule Endoscopy

Capsule Endoscopy (CE) is a non-invasive medical imaging technique that utilizes a small wireless camera embedded in a capsule to capture images of the digestive tract. This innovative technology allows for a thorough examination of the gastrointestinal system, aiding in the detection and diagnosis of various conditions. CE is particularly useful in visualizing areas that are challenging to reach with traditional endoscopy methods.





![image](https://github.com/A-dvika/Capsule_Endoscopy_mini_project/assets/115079077/86a06aa8-886d-4152-a690-1cd27929f684)


![image](https://github.com/A-dvika/Capsule_Endoscopy_mini_project/assets/115079077/641df251-a730-4841-9ffc-12bd7bbaf6e4)




## Overview
This project focuses on leveraging Artificial Intelligence (AI) for bleeding detection in Capsule Endoscopy (CE) videos. The AI model is trained on a dataset comprising 2618 annotated frames, encompassing various gastrointestinal bleeding instances. The goal is to enhance the efficiency, accuracy, and accessibility of bleeding detection in the context of non-invasive medical imaging.

## Key Features
### Early Detection: 
The AI model enables early identification of gastrointestinal bleeding, facilitating timely medical intervention.

### Non-Invasive Approach: 
Capitalizing on the non-invasive nature of Capsule Endoscopy, the project offers a patient-friendly diagnostic process.

### Efficiency and Speed:
Rapid processing of large CE video datasets for quick analysis of potential bleeding instances.

### Enhanced Accuracy:
Utilizing machine learning algorithms to improve the accuracy of bleeding detection, minimizing false positives/negatives.

### Generalization and Vendor Independence:
The project aims for broad applicability, ensuring the model can be generalized across different scenarios and equipment.

### Resource Optimization: 
Automated bleeding detection allows healthcare providers to optimize resources and focus human expertise on complex cases.

## How It Helps
### Improved Patient Outcomes: 
Early detection contributes to better patient outcomes by enabling timely medical intervention.

### Efficient Diagnostic Process: 
The project enhances the efficiency of the diagnostic process through automated bleeding detection.

### Non-Invasive Diagnostic Experience: 
Patients benefit from a more comfortable and less invasive diagnostic experience compared to traditional endoscopic methods.

### Optimized Resource Allocation: 
Healthcare providers can optimize resources by automating the initial screening process, directing human expertise where it's needed most.

### Contribution to Research: 
The project's datasets contribute to ongoing research in medical imaging and AI, fostering advancements in understanding and addressing gastrointestinal conditions.

# MedNet: Medical Image Classification and Segmentation

## Overview

This project implements a neural network architecture, called ClassifyViStA, designed for medical image classification and segmentation tasks. The architecture combines features from ResNet-18, VGG16, and a U-net styled decoder. It performs simultaneous image classification and segmentation on medical images, leveraging both global and local features.

## Architecture

### 1. ResNet-18 Model (CustomResNet18WithMask)

- **Backbone:**
  - ResNet-18 for feature extraction.
  - Modified final fully connected layer for binary classification.

- **Additional Layers:**
  - Convolutional layer (`center`) for additional processing.
  - U-net styled decoder (`Decoder` class) for segmentation.

- **Output:**
  - Predictions for classification (`img_output`) and segmentation (`seg`).

### 2. VGG16 Model (CustomVGGWithMask)

- **Backbone:**
  - VGG16 for feature extraction.
  - Modified final fully connected layer for binary classification.

- **Additional Layers:**
  - Convolutional layer (`center`) for additional processing.
  - U-net styled decoder (`Decoder` class) for segmentation.

- **Output:**
  - Predictions for classification (`img_output`) and segmentation (`seg`).

### 3. Common Elements

- Both models take input images and masks.
- Masks are downsampled for segmentation enhancement.
- Segmentation branch includes decoding layers for gradual upsampling.

## Training

- Loss function: Binary cross-entropy and a custom hybrid segmentation loss.
- Data augmentation techniques: flipping, rotation, and blurring.
- Training loop iterates over epochs, updating parameters using SGD.
- Evaluation metrics: Accuracy, precision, recall, and F1-score.

# YOLOv8 Bleeding Frames Detection

code for detecting bleeding frames in medical images using YOLOv8, a state-of-the-art object detection model. The trained model is capable of identifying regions of interest indicating potential bleeding in medical images.

## Features

- **Object Detection:** Utilizes YOLOv8 for accurate and efficient object detection.
- **Bleeding Frames:** Specifically trained to detect bleeding frames in medical images.
- **Configurable Options:** Adjustable parameters for fine-tuning detection performance.

## Usage

1. Clone the repository: `git clone https://github.com/yourusername/your-repo.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run bleeding frames detection: `python detect_bleeding.py --source /path/to/images`

## Pre-trained Model

The YOLOv8 model for bleeding frames detection is pre-trained and included in the `pretrained_models` directory. 

