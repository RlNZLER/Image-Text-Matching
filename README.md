# Advanced Machine Learning
## Task 1
You are required to use Machine Learning to tackle the problem of **“Image-Text Matching”**. Given an image and a textual description, the task is to predict whether they match or not. This implies a binary classification task, where there is a match when the image and text correspond to each other—no match otherwise, see examples in the table below. The data for this task will be generated by the delivery team from the Flickr8k or Flicker30k datasets. This data was originally proposed by (Hodosh et al, 2013, JAIR) and (Young et al, 2014, TACL) to investigate methods at the intersection between machine learning, computer vision, and natural language processing.

| Image | Text | Label |
| ----- | ---- | ----- |
| ![Image Description](https://github.com/RlNZLER/AML_Task1/blob/main/data/images/3523972229_d44e9ff6d7.jpg?raw=true) | The skateboard is falling away from the skateboarder as he attempts a jump. | match |
| ![Image Description](https://github.com/RlNZLER/AML_Task1/blob/main/data/images/3524436870_7670df68e8.jpg?raw=true) | Two people stand next to a sign that says  HELL  made out of stars. | no-match |
| ![Image Description](https://github.com/RlNZLER/AML_Task1/blob/main/data/images/3525453732_f74a38f111.jpg?raw=true) | An owl has its wings up and widespread during the day. | match |
| ![Image Description](https://github.com/RlNZLER/AML_Task1/blob/main/data/images/3526897578_3cf77da99b.jpg?raw=true) | A kayaker is battling waves in a purple boat while wearing rain gear. | no-match |

Your task involves training and evaluating at least three machine-learning classifiers for matching images and sentences. To do that, you will use a unique dataset (different and potentially smaller than the originally proposed due to computing requirements) available via Blackboard. Since the dataset will contain different data splits (training, validation, and test examples), you are reminded that the test data should not be involved during training. It should only be used for testing your trained classifiers, and it should be used only to report the performance of the predictions of your proposed models.

You are expected to explore a range of machine learning classifiers inspired by the materials presented and discussed within the module and beyond (i.e., from reading related literature). You will investigate compare and critique their performance to justify your recommendations. Your comparisons should include metrics such as Balanced Classification Accuracy, precision-recall, F-measure, and any other metrics that you consider relevant. In this assignment, you are free to train any classifier, do any pre-and/or post-processing of the data, and implement your algorithm(s) whenever possible instead of only using libraries. You are allowed to use publicly available libraries to train your models; for example, you can use any deep learning frameworks and tools (e.g. Keras, Tensorflow, Pytorch, ChatGPT). However, you will need to clearly mention your resources, acknowledge them appropriately, and compare between classifiers and their results in your report.

**Store requirements:**
pip freeze > requirements.txt

**Download and install CUDA**
https://youtu.be/4p4b9qT5S6w?si=tT6QEPMxwRY-fR0m


## Image-Text Matching Classifier: Assignment Task 1

### Overview

To tackle Task 1 of your assignment, you'll need to follow these steps:

1. **Data Preparation**:
   - Obtain the dataset provided by the delivery team, containing image-text pairs with corresponding labels.
   - Split the dataset into training, validation, and test sets.

2. **Feature Extraction/Engineering**:
   - Extract features from both images and text. For images, this might involve using pre-trained convolutional neural networks (CNNs) like VGG, ResNet, or Inception to extract image features. For text, techniques like word embeddings (e.g., Word2Vec, GloVe) or transformer models (e.g., BERT) can be used.
   - Engineer additional features if necessary, such as image and text length, or any domain-specific features.

3. **Model Selection and Training**:
   - Choose at least three machine learning classifiers to train on the extracted features. Some options include Support Vector Machines (SVM), Random Forests, Logistic Regression, Gradient Boosting Machines (GBM), or Neural Networks.
   - Train the chosen classifiers on the training set using the extracted features.

4. **Evaluation**:
   - Evaluate the trained classifiers on the validation set using metrics such as Balanced Classification Accuracy, Precision-Recall, F-measure, etc.
   - Tune hyperparameters if necessary based on validation performance.

5. **Comparison and Critique**:
   - Compare the performance of the trained classifiers on the validation set.
   - Critique the strengths and weaknesses of each classifier, considering factors such as interpretability, computational complexity, and generalization ability.

6. **Testing**:
   - Select the best-performing classifier based on validation performance.
   - Evaluate this classifier on the test set and report its performance using the agreed-upon metrics.

7. **Documentation**:
   - Clearly document the entire process in a report, including data preprocessing steps, feature extraction methods, model architectures, training procedures, evaluation metrics, results, comparisons, and critiques.
   - Mention any publicly available libraries or tools used for training the models.
   - Acknowledge appropriate sources and references.
  
## Baseline System for Image-Text Matching

The provided Python script contains a baseline system for the Image-Text Matching task. Let's break down the main components and functionalities:

1. **Dependencies and Imports**:
   - The script imports necessary libraries and dependencies such as TensorFlow, NumPy, and others required for data processing, model building, and evaluation.

2. **Data Loading**:
   - The `ITM_DataLoader` class is responsible for loading and preprocessing the data. It loads image data, text embeddings, and corresponding labels from files. It also provides methods for processing input data and creating TensorFlow datasets.

3. **Model Building**:
   - The `ITM_Classifier` class builds the main deep learning model for the Image-Text Matching task. It consists of separate branches for processing images and text embeddings. These branches are then concatenated to form a multimodal model.
   - The model architecture includes convolutional layers for image processing and dense layers for text embeddings. It uses dropout for regularization and layer normalization for stabilizing training.

4. **Model Training**:
   - The script trains the model using the training dataset and validates it using the validation dataset. It utilizes TensorFlow's `fit` method for training.
   - It also implements custom learning rate scheduling using the `optimization.create_optimizer` function from TensorFlow's official library.

5. **Model Evaluation**:
   - After training, the model is evaluated on the test dataset. It predicts image-text matches and computes accuracy metrics.
   - Both custom evaluation logic and TensorFlow's built-in evaluation methods are used to assess the model's performance.

6. **Output and Results**:
   - The script outputs sample predictions during testing, showing the predicted match probabilities and corresponding image filenames.
   - It reports the test accuracy obtained from both custom calculations and TensorFlow's evaluation method.

7. **Usage and Instructions**:
   - Instructions for running the script are provided, including dependencies to be installed and steps to execute the program in different environments (e.g., Windows, Linux).
   - Users are encouraged to modify and extend the baseline system according to their needs and explore alternative solutions.

Overall, the script serves as a starting point for implementing an Image-Text Matching classifier using deep learning techniques and TensorFlow. Users can further enhance the system by experimenting with different architectures, hyperparameters, and training strategies.

https://arxiv.org/pdf/1909.11740v3.pdf
https://github.com/ChenRocks/UNITER


Certainly! Hyperparameters are parameters whose values are set before the training process begins and remain constant during training. Tuning these hyperparameters can significantly impact the performance of your model. Here are the hyperparameters you can tweak to improve the accuracy of your image-text matching model:

1. **Learning Rate (`learning_rate`):**
   - Learning rate determines the step size taken during optimization. Too high a learning rate may cause the model to overshoot the optimal solution, while too low a learning rate may result in slow convergence.
   - Try different learning rates (e.g., 1e-2, 1e-3, 1e-4) and monitor the training curves to find the optimal value.

2. **Number of Epochs (`epochs`):**
   - Epochs refer to the number of times the entire dataset is passed through the model during training.
   - Increasing the number of epochs may allow the model to converge to a better solution, but be cautious of overfitting.
   - Experiment with different numbers of epochs and use early stopping to prevent overfitting.

3. **Batch Size (`BATCH_SIZE`):**
   - Batch size determines the number of samples processed in each iteration during training.
   - Larger batch sizes may lead to faster convergence but require more memory. Smaller batch sizes may offer better generalization.
   - Try different batch sizes (e.g., 16, 32, 64) and choose based on the trade-off between convergence speed and memory usage.

4. **Optimizer Type (`optimizer_type`):**
   - The choice of optimizer affects how the model updates its parameters during training.
   - Common optimizers include Adam, SGD, RMSprop, etc.
   - Experiment with different optimizers and their parameters to find the one that works best for your model and dataset.

5. **Dropout Rate (`dropout_rate`):**
   - Dropout is a regularization technique that randomly drops a fraction of neurons during training to prevent overfitting.
   - Adjusting the dropout rate controls the strength of regularization.
   - Try different dropout rates (e.g., 0.1, 0.2, 0.5) and choose based on the model's performance on the validation set.

6. **Projection Layers and Dimensions (`num_projection_layers`, `projection_dims`):**
   - These hyperparameters control the architecture of the projection layers used in the image and text encoders.
   - Experiment with the number of projection layers and their dimensions to find the optimal balance between model capacity and generalization.

7. **Warmup Steps (`num_warmup_steps`):**
   - Warmup steps gradually increase the learning rate at the beginning of training to stabilize the optimization process.
   - Adjust the number of warmup steps based on the dataset size and model complexity.

8. **Model Architecture:**
   - Explore different architectures for image and text encoders, such as varying the number of convolutional layers, kernel sizes, or using pre-trained models for feature extraction.
   - Experiment with attention mechanisms or transformer-based architectures for capturing long-range dependencies in text data.

9. **Data Augmentation (`data augmentation`):**
   - Apply data augmentation techniques to increase the diversity of training samples and improve the model's robustness.
   - Common data augmentation techniques for images include random rotation, translation, scaling, etc.

10. **Early Stopping (`callbacks`):**
    - Early stopping prevents overfitting by monitoring the validation loss and stopping training when performance on the validation set starts to degrade.
    - Tune the patience parameter to control the number of epochs to wait before stopping.

11. **Regularization (`regularization`):**
    - Explore other regularization techniques such as L1 or L2 regularization to penalize large weights and prevent overfitting.

Experimenting with these hyperparameters and monitoring the model's performance on validation data is essential for finding the optimal configuration for your image-text matching model.