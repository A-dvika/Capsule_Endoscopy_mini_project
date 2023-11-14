

# Capsule Endoscopy

Capsule Endoscopy (CE) is a non-invasive medical imaging technique that utilizes a small wireless camera embedded in a capsule to capture images of the digestive tract. This innovative technology allows for a thorough examination of the gastrointestinal system, aiding in the detection and diagnosis of various conditions. CE is particularly useful in visualizing areas that are challenging to reach with traditional endoscopy methods. 
![image](https://github.com/A-dvika/Capsule_Endoscopy_mini_project/assets/115079077/86a06aa8-886d-4152-a690-1cd27929f684)


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
## Evaluation Metrics

 ### Classification Metrics
| Metric (in probability)| Value    |
|------------------------|----------|
| Accuracy               |   0.4937 |
| Recall                 |   0.5753 |
| F1-Score               |   0.661  |


### Detection Metrics
| Metric (in probability)| Value          |
|------------------------|----------------|
| Average Precision      |     0.5081     |
| Mean Average Precision |     0.652      |
| Intersection over Union|     0.4937     |

# Validation Dataset
## Detection and Classification

| **Imagename** | **img- (271).png** | **img- (386).png**|**img- (389).png**|**img- (406).png**|**img- (409).png**|
|------ |---------------------|---------------------|---------------------|---------------------|---------------------|
|**Images** | <img src="Images_README\validation_dataset\classification_and_detection\img- (271).png" alt="Image 1">| <img src="Images_README\validation_dataset\classification_and_detection\img- (386).png" alt="Image 1">|<img src="Images_README\validation_dataset\classification_and_detection\img- (389).png" alt="Image 1">| <img src="Images_README\validation_dataset\classification_and_detection\img- (406).png" alt="Image 1">|<img src="Images_README\validation_dataset\classification_and_detection\img- (409).png" alt="Image 1">|
|**Confidance**| 0.96 | 0.96 |0.96 | 0.96 |0.96 |
                                                                                                         

| **Imagename** | **img- (608).png** | **img- (609).png**|**img- (797).png**|**img- (908).png**|**img- (912).png**|
|------ |---------------------|---------------------|---------------------|---------------------|---------------------|
|**Images** | <img src="Images_README\validation_dataset\classification_and_detection\img- (608).png" alt="Image 1">| <img src="Images_README\validation_dataset\classification_and_detection\img- (609).png" alt="Image 1">|<img src="Images_README\validation_dataset\classification_and_detection\img- (797).png" alt="Image 1">| <img src="Images_README\validation_dataset\classification_and_detection\img- (908).png" alt="Image 1">|<img src="Images_README\validation_dataset\classification_and_detection\img- (912).png" alt="Image 1">|
|**Confidance**| 0.96 | 0.96 |0.97 | 0.97 |0.97 |

## Interpretability Plots (Cam Plots of 2nd last layer)

| **Imagename** | **img- (271).png** | **img- (386).png**|**img- (389).png**|**img- (406).png**|**img- (409).png**|
|------ |---------------------|---------------------|---------------------|---------------------|---------------------|
|**Images** | <img src="Images_README\validation_dataset\interpretability_plots\img- (271)_cam.png" alt="Image 1">| <img src="Images_README\validation_dataset\interpretability_plots\img- (386)_cam.png" alt="Image 1">|<img src="Images_README\validation_dataset\interpretability_plots\img- (389)_cam.png" alt="Image 1">| <img src="Images_README\validation_dataset\interpretability_plots\img- (406)_cam.png" alt="Image 1">|<img src="Images_README\validation_dataset\interpretability_plots\img- (409)_cam.png" alt="Image 1">|
                                                                                                         
| **Imagename** | **img- (608).png** | **img- (609).png**|**img- (797).png**|**img- (908).png**|**img- (912).png**|
|------ |---------------------|---------------------|---------------------|---------------------|---------------------|
|**Images** | <img src="Images_README\validation_dataset\interpretability_plots\img- (608)_cam.png" alt="Image 1">| <img src="Images_README\validation_dataset\interpretability_plots\img- (609)_cam.png" alt="Image 1">|<img src="Images_README\validation_dataset\interpretability_plots\img- (797)_cam.png" alt="Image 1">| <img src="Images_README\validation_dataset\interpretability_plots\img- (908)_cam.png" alt="Image 1">|<img src="Images_README\validation_dataset\interpretability_plots\img- (912)_cam.png" alt="Image 1">|


# Testing Dataset 1
## Detection and Classification

| **Imagename** | **A0001.png** | **A0033.png**|**A0035.png**|**A0040.png**|**A0041.png**|
|------ |---------------------|---------------------|---------------------|---------------------|---------------------|
|**Images** | <img src="Images_README\testing_dataset_1\Classification_and_detection\A0001.png" alt="Image 1">| <img src="Images_README\testing_dataset_1\Classification_and_detection\A0033.png" alt="Image 1">|<img src="Images_README\testing_dataset_1\Classification_and_detection\A0035.png" alt="Image 1">| <img src="Images_README\testing_dataset_1\Classification_and_detection\A0040.png" alt="Image 1">|<img src="Images_README\testing_dataset_1\Classification_and_detection\A0041.png" alt="Image 1">|
|**Confidance**| 0.29 | 0.75 |0.44 | 0.37 | 0.27 |

## Interpretability Plots (Cam Plots of 2nd last layer)                                                                                                         
| **Imagename** | **A0001_cam.png** | **A0033_cam.png**|**A0035_cam.png**|**A0040_cam.png**|**A0041_cam.png**|
|------ |---------------------|---------------------|---------------------|---------------------|---------------------|
|**Images** | <img src="Images_README\testing_dataset_1\Interpretability_plots\A0001_cam.png" alt="Image 1">| <img src="Images_README\testing_dataset_1\Interpretability_plots\A0033_cam.png" alt="Image 1">|<img src="Images_README\testing_dataset_1\Interpretability_plots\A0035_cam.png" alt="Image 1">| <img src="Images_README\testing_dataset_1\Interpretability_plots\A0040_cam.png" alt="Image 1">|<img src="Images_README\testing_dataset_1\Interpretability_plots\A0041_cam.png" alt="Image 1">|

# Testing Dataset 2
## Detection and Classification

| **Imagename** | **A0211.png** | **A0498.png**|**A0500.png**|**A0532.png**|**A0551.png**|
|------ |---------------------|---------------------|---------------------|---------------------|---------------------|
|**Images** | <img src="Images_README\testing_dataset_2\classification_and_detection\A0211.png" alt="Image 1">| <img src="Images_README\testing_dataset_2\classification_and_detection\A0498.png" alt="Image 1">|<img src="Images_README\testing_dataset_2\classification_and_detection\A0500.png" alt="Image 1">| <img src="Images_README\testing_dataset_2\classification_and_detection\A0532.png" alt="Image 1">|<img src="Images_README\testing_dataset_2\classification_and_detection\A0551.png" alt="Image 1">|
|**Confidance**| 0.27 | 0.28 |0.62 | 0.27 |0.32 |

## Interpretability Plots (Cam Plots of 2nd last layer)                                                                                                         
| **Imagename** | **A0211_cam.png** | **A0498_cam.png**|**A0500_cam.png**|**A0532_cam.png**|**A0551_cam.png**|
|------ |---------------------|---------------------|---------------------|---------------------|---------------------|
|**Images** | <img src="Images_README\testing_dataset_2\Interpretability_plots\A0211_cam.png" alt="Image 1">| <img src="Images_README\testing_dataset_2\Interpretability_plots\A0498_cam.png" alt="Image 1">|<img src="Images_README\testing_dataset_2\Interpretability_plots\A0500_cam.png" alt="Image 1">| <img src="Images_README\testing_dataset_2\Interpretability_plots\A0532_cam.png" alt="Image 1">|<img src="Images_README\testing_dataset_2\Interpretability_plots\A0551_cam.png" alt="Image 1">|

# Training Procedure
Using training.ipynb and config.yaml, you can train model using train_model function which takes model destination path (model_path), config.yaml file (config_yaml), epochs(total_epochs), image size (image_size) and device type (device_) as input and trains the model.

**Note:** Please give full path of you dataset in config.yaml instead of giving relative path.

# Prediction procedure
For prediction use function prediction which is present in prediction.ipynb. This function takes the model destination path, test data directory path as input, and returns boundary box parameters as output.
