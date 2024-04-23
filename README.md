# Image-Text Matching Task

### Overview
--------

This repository contains resources and implementation details for the task of Image-Text Matching as part of the Advanced Machine Learning course. The goal of this task is to develop machine learning models that can accurately predict whether a given image and text description match or not, based on a binary classification approach.

### Dataset
-------

The dataset utilized for this task is derived from the Flickr8k and Flickr30k datasets encompassing separate training, validation, and test splits.

### Implemented Classifiers

1.  **ResNet50 with BERT:** Combines the power of ResNet50 for image feature extraction with BERT for text feature extraction.
2.  **Inception V3 with BERT:** Utilizes Inception V3 for image processing paired with BERT for processing the textual data.
3.  **InceptionResNetV2 with BERT:** Integrates InceptionResNetV2 for advanced image feature extraction with BERT for textual analysis.
   
### Task Description
----------------

Participants are required to train and evaluate at least three different machine learning classifiers to determine if an image and its corresponding textual description match. The evaluation of these classifiers should include metrics such as Balanced Classification Accuracy, precision-recall, and F-measure, among others.

| Image | Text | Label |
| ----- | ---- | ----- |
| ![Image Description](https://github.com/RlNZLER/AML_Task1/blob/main/images/3523972229_d44e9ff6d7.jpg?raw=true) | The skateboard is falling away from the skateboarder as he attempts a jump. | match |
| ![Image Description](https://github.com/RlNZLER/AML_Task1/blob/main/images/3524436870_7670df68e8.jpg?raw=true) | Two people stand next to a sign that says  HELL  made out of stars. | no-match |
| ![Image Description](https://github.com/RlNZLER/AML_Task1/blob/main/images/3525453732_f74a38f111.jpg?raw=true) | An owl has its wings up and widespread during the day. | match |
| ![Image Description](https://github.com/RlNZLER/AML_Task1/blob/main/images/3526897578_3cf77da99b.jpg?raw=true) | A kayaker is battling waves in a purple boat while wearing rain gear. | no-match |


### Installation Instructions
-------------------------

#### Step 1: Install Anaconda

+ Download and install Anaconda to manage packages and environments for the project.
+ Create a new conda environment with **Python 3.10**.
+ Launch the conda terminal of the newly created environment.

#### Step 2: Set up the environment

Install CUDA and cuDNN libraries required for running TensorFlow with GPU support.

```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

Note: TensorFlow versions above 2.10 are not supported on native GPU on Windows.

Install TensorFlow 2.10:

```
python -m pip install "tensorflow<2.11"
```

Verify the GPU setup:

```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```


#### Step 3: Install required Python packages

```
pip install -r ./requirements.txt
```

### Usage
-----

After installation, you can run the scripts provided in the repository under src to train models and evaluate their performance. Detailed usage instructions are provided in the respective script documentation.

### Contributions
-------------

Contributions are welcome. Please ensure to follow the best practices and guidelines for pull requests.

### Acknowledgments
---------------

This project uses publicly available libraries and frameworks such as Keras, TensorFlow, PyTorch, and ChatGPT. Proper acknowledgment to the data sources and tools used is crucial for transparency and reproducibility.

### License
-------
