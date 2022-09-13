# Master Thesis: Pedestrian detection and tracking for safe robot navigation in urban environments

## **Abstract**
Data-driven methods for tracking people’s poses and predicting their actions based on video streams have been successfully applied in numerous fields, including
surveillance, service robotics, rehabilitation and healthcare physiotherapy, to name a few. However, the literature falls short of methods that attempt to apply these
concepts in the context of autonomous outdoor navigation. More precisely, no one has yet attempted to combine these concepts into an end-to-end solution for
real-time skeleton-based action recognition that supports the decision-making process for the task of autonomous outdoor navigation in urban areas. As such, this
work aims to do just that, using the Lightweight OpenPose architecture for pose estimation and ST-GCN architecture for action classification and testing its realtime capabilities on a NVIDIA Jetson Xavier NX. In addition, the end-to-end solution will be extended with gaze estimation and proximity measurement algorithms to obtain more information about the outside world. This will provide the autonomous robots with a lot of information that will help them in the future with path planning and decision making.

### **About the project**
This project was a joint cooperation between [Capra Robotics ApS](https://capra.ooo/) and the Faculty of the [Technical University of Denmark](https://www.dtu.dk/)  in partial fulfillment of the requirements for the academic degree of Master of Science, studyline [autonomous systems](https://www.dtu.dk/english/education/msc/programmes/autonomous-systems#study-programme__curriculum).

### **Authors**

* [David Anthony Parham](https://github.com/davelbit) (s202385)

### **Acknowledgments**
The provided source course is based on the [OpenDR Project](https://github.com/opendr-eu/opendr), however it only contains the necessary code for the end-end-solution.

### **End-to-end solution**
The proposed solution consists of two deep neural network models (OpenPose (lightweight) and ST-GCN) and two additional algorithms for head pose estimation and proximity measurement.
As illustrated in the figure below.

<img src="media/Flow_data_diagram1.png" alt="Confusion_Matrix_Siamese" width="640"/>


### **Methods explained**

#### **Human Pose Estimation**

#### **Action Recognition**

#### **Head Pose Estimation**

#### **Proximity measurement**

### **Output**

#### **Visual Results**

True Positives          |  False Positives
:-------------------------:|:-------------------------:
<img src="media/true_positive_1.png" alt="1st True Positive" width="640"/>  |  <img src="media/false_positive_1.png" alt="1st False Positive" width="640"/>
<img src="media/true_positive_2.png" alt="2nd True Positive" width="640"/>  |  <img src="media/false_positive_2.png" alt="2nd False Positive" width="640"/>

>Note: The confidence score reflects to which extend the network is confident that those two images belong to the same class. High -> Similar; Low -> Dissimilar.


#### **Performance Table**

<img src="media/Model-Comparison-1.png" alt="Model-Comparison_table" width="640"/>

> Note: The Siamese Network used resnet34 as backbone, whereas the Prototypical Network was using resnet18.


### **Installation**

Please refer to the official [installation](https://github.com/opendr-eu/opendr/blob/master/docs/reference/installation.md) guidelines of the OpenDr project.

Afterwards run the following command line command to ensure that all needed dependencies are installed.

To install the required libraries, run:
```bash
$ pip install -r requirements.txt
```

### **Project structure**

#### **General**

- [/data/](https://github.com/davelbit/DTU-Object-Detection-via-Few-Shot-Learning/tree/main/data)
    - Training and testing datasets.

#### **Siamese**

- [/Siamese_torch/](https://github.com/davelbit/DTU-Object-Detection-via-Few-Shot-Learning/tree/main/Siamese_torch)
    - Necessary code to train and evaluate the
Siamese Network for Few-Shot Learning.
- [/Siamese_torch/core/siamese_network.py](https://github.com/davelbit/DTU-Object-Detection-via-Few-Shot-Learning/blob/main/Siamese_torch/core/siamese_network.py)
    - Network architecture.
- [/Siamese_torch/core/config.py](https://github.com/davelbit/DTU-Object-Detection-via-Few-Shot-Learning/blob/main/Siamese_torch/core/config.py)
    - Configuration file.
- [/Siamese_torch/core/utils.py](https://github.com/davelbit/DTU-Object-Detection-via-Few-Shot-Learning/blob/main/Siamese_torch/core/utils.py)
    - Loss functions, Confussion_Matrix, 3D-Scatter-Plot, etc.
- [/Siamese_torch/output](https://github.com/davelbit/DTU-Object-Detection-via-Few-Shot-Learning/tree/main/Siamese_torch/output)
    - Model checkpoints, saved_models and prediction_images.
- [/Siamese_torch/datasets.py](https://github.com/davelbit/DTU-Object-Detection-via-Few-Shot-Learning/blob/main/Siamese_torch/datasets.py)
    - Custom dataloader to process our datasets.
- [/Siamese_torch/split_dataset.py](https://github.com/davelbit/DTU-Object-Detection-via-Few-Shot-Learning/blob/main/Siamese_torch/split_dataset.py)
    - Preprocessing and creation of few-shot learning datasets.
- [/Siamese_torch/train_siamese_torch.py](https://github.com/davelbit/DTU-Object-Detection-via-Few-Shot-Learning/blob/main/Siamese_torch/train_siamese_torch.py)
    - Main code to run the training process.
- [/Siamese_torch/test_network.py](https://github.com/davelbit/DTU-Object-Detection-via-Few-Shot-Learning/blob/main/Siamese_torch/test_network.py)
    - Main code to run the evaluation process.

**Prototypical**
- [/Prototypical_torch/](https://github.com/davelbit/DTU-Object-Detection-via-Few-Shot-Learning/blob/main/Prototypical_torch/prototypical_torch/)
    - Necessary code to train and evaluate the
Prototypical Network for Few-Shot Learning.
- [/Prototypical_torch/prototypical_model.py](https://github.com/davelbit/DTU-Object-Detection-via-Few-Shot-Learning/blob/main/Prototypical_torch/prototypical_torch/prototypical_model.py)
    - Network architecture.
- [/Prototypical_torch/resnet_train.py](https://github.com/davelbit/DTU-Object-Detection-via-Few-Shot-Learning/blob/main/Prototypical_torch/prototypical_torch/resnet_train.py)
    - Main code to run the training process.
- [/Prototypical_torch/datasets.py](https://github.com/davelbit/DTU-Object-Detection-via-Few-Shot-Learning/blob/main/Prototypical_torch/prototypical_torch/datasets.py)
    - Custom dataloader to process our datasets.
