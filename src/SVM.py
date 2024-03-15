#####################################################
# Image-Text Matching Classifier: baseline system
# 
# This program has been adapted and rewriten from sources such as:
# https://www.tensorflow.org/tutorials/images/cnn
# https://keras.io/api/layers/merging_layers/concatenate/
# https://pypi.org/project/sentence-transformers/
#
# If you are new to Tensorflow, read the following brief tutorial:
# https://www.tensorflow.org/tutorials/quickstart/beginner
#
# As you develop your experience and skills, you may want to check
# details of particular aspects of the Tensorflow API: 
# https://www.tensorflow.org/api_docs/python/tf/all_symbols
#
# This is a binary classifier for image-text matching, where the inputs
# are images and text-based features, and the outputs (denoted as
# match=[1,0] and nomatch=[0,1]) correspond to (predicted) answers. This 
# baseline classifier makes use of two strands of features. The first 
# are produced by a CNN-classifier, and the second are derived offline 
# from a sentence embedding generator. The latter have the advantage of
# being generated once, which can accelerate training due to being 
# pre-trained and loaded at runtime. Those two strands of features are
# concatenated at trianing time to form a multimodal set of features, 
# combining learnt image features and pre-trained sentence features.

# This program has been tested using an Anaconda environment with Python 3.9
# and 3.10 on Windows 11 and Linux Ubuntu 22. The easiest way to run this 
# baseline at Uni is by booting your PC with Windows and using the following steps.
#
# Step 1=> Make sure that your downloaded data and baseline system are 
#          extracted in the Downloads folder.
#          Note. Your path should start with /mnt/c/Users/Computing/Downloads
#
# Step 2=> Open a terminal and select Ubuntu from the little arrow pointing down
#          Note. Your program will be executed under a Linux environment.
#
# Step 3=> Install the following dependencies:
# pip install tf-models-official
# pip install tensorflow-text
# pip install einops
#
# Step 4=> Edit file ITM_Classifier-baseline.py and make sure that variable 
# IMAGES_PATH points to the right folder containing the data.
#
# Step 5=> Run the program using a command such as 
# $ python ITM_Classifier-baseline.py
#
#
# The code above can also be run from Visual Studio Code, to access it using the
# Linux envinment type "code ." in the Ubuntu terminal. From VSCode, click View, Terminal, 
# type your command (example: python ITM_Classifier-baseline.py) and Enter.
#
# Running this baseline takes about 5 minutes minutes with a GPU-enabled Uni PC.
# WARNING: Running this code without a GPU is too slow and not recommended.
# 
# In your own PC you can use Anaconda to run this code. From a conda terminal 
# for example. If you want GPU-enabled execution, it is recommended that you 
# install the following versions of software:
# CUDA 11.8
# CuDNN 8.6
# Tensorflow 2.10
#
# Feel free to use and/or modify this program as part of your CMP9137 assignment.
# You are invited to use the knowledge acquired during lectures, workshops 
# and beyond to propose and evaluate alternative solutions to this baseline.
# 
# Version 1.0, main functionality tested with COCO data 
# Version 1.2, extended functionality for Flickr data
# Contact: {hcuayahuitl, lzhang, friaz}@lincoln.ac.uk
#####################################################


# Let's import the dependencies

import sys
import os
import time
import einops
import pickle
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from official.nlp import optimization
import matplotlib.pyplot as plt

# Verify TensorFlow can detect the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"Num GPUs Available: {len(gpus)}")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if len(gpus) > 0:
    print(f"Using GPU: {gpus[0].device_type} {gpus[0].name}")
else:
    print("No GPU detected. Running on CPU.")
    
# Class for loading image and text data

class ITM_DataLoader():
    BATCH_SIZE = 16
    IMAGE_SIZE = (224, 224)
    IMAGE_SHAPE = (224, 224, 3)
    SENTENCE_EMBEDDING_SHAPE = (384)
    AUTOTUNE = tf.data.AUTOTUNE
    IMAGES_PATH = "data/images"
    train_data_file = IMAGES_PATH+"/../flickr8k.TrainImages.txt"
    dev_data_file = IMAGES_PATH+"/../flickr8k.DevImages.txt"
    test_data_file = IMAGES_PATH+"/../flickr8k.TestImages.txt"
    sentence_embeddings_file = IMAGES_PATH+"/../flickr8k.cmp9137.sentence_transformers.pkl"
    sentence_embeddings = {}
    train_ds = None
    val_ds = None
    test_ds = None

    def __init__(self):
        self.sentence_embeddings = self.load_sentence_embeddings()
        self.train_ds = self.load_classifier_data(self.train_data_file)
        self.val_ds = self.load_classifier_data(self.dev_data_file)
        self.test_ds = self.load_classifier_data(self.test_data_file)
        print("done loading data...")

    # Sentence embeddings are dense vectors representing text data, one vector per sentence. 
    # Sentences with similar vectors would mean sentences with equivalent meanning.  
	# They are useful here to provide text-based features of questions in the data.
    # Note: sentence embeddings don't include label info, they are solely based on captions.
    def load_sentence_embeddings(self):
        sentence_embeddings = {}
        print("READING sentence embeddings...")
        with open(self.sentence_embeddings_file, 'rb') as f:
            data = pickle.load(f)
            for sentence, dense_vector in data.items():
                sentence_embeddings[sentence] = dense_vector
                #print("*sentence=",sentence)
        print("Done reading sentence_embeddings!")
        return sentence_embeddings

    # In contrast to text-data based on pre-trained features, image data does not use
    # any form of pre-training in this program. Instead, it makes use of raw pixels.
    # Notes that input features to the classifier are only pixels and sentence embeddings.
    def process_input(self, img_path, dense_vector, text, label):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.IMAGE_SIZE)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.cast(img, tf.float32) / 255
        features = {}
        features["image_input"] = img
        features["text_embedding"] = dense_vector
        features["caption"] = text
        features["file_name"] = img_path
        return features, label

    # This method loads the multimodal data, which comes from the following sources:
    # (1) image files in IMAGES_PATH, and (2) files with pattern flickr8k.*Images.txt
    # The data is stored in a tensorflow data structure to make it easy to use by
    # the tensorflow model during training, validation and test. This method was 
    # carefully prepared to load the data rapidly, i.e., by loading already created
    # sentence embeddings (text features) rather than creating them at runtime.
    def load_classifier_data(self, data_files):
        print("LOADING data from "+str(data_files))
        print("=========================================")
        image_data = []
        text_data = []
        embeddings_data = []
        label_data = []
		
        # get image, text, label of image_files
        with open(data_files) as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip("\n")
                img_name, text, raw_label = line.split("	")
                img_name = os.path.join(self.IMAGES_PATH, img_name.strip())

                # get binary labels from match/no-match answers
                label = [1, 0] if raw_label == "match" else [0, 1]
                #print("I=%s T=%s _L=%s L=%s" % (img_name, text, raw_label, label)) 

				# get sentence embeddings (of textual captions)
                text_sentence_embedding = self.sentence_embeddings[text]
                text_sentence_embedding = tf.constant(text_sentence_embedding)

                image_data.append(img_name)
                embeddings_data.append(text_sentence_embedding)
                text_data.append(text)
                label_data.append(label)

        print("|image_data|="+str(len(image_data)))
        print("|text_data|="+str(len(text_data)))
        print("|label_data|="+str(len(label_data)))
		
        # prepare a tensorflow dataset using the lists generated above
        dataset = tf.data.Dataset.from_tensor_slices((image_data, embeddings_data, text_data, label_data))
        dataset = dataset.shuffle(self.BATCH_SIZE * 8)
        dataset = dataset.map(self.process_input, num_parallel_calls=self.AUTOTUNE)
        dataset = dataset.batch(self.BATCH_SIZE).prefetch(self.AUTOTUNE)
        self.print_data_samples(dataset)
        return dataset

    def print_data_samples(self, dataset):
        print("PRINTING data samples...")
        print("-----------------------------------------")
        for features_batch, label_batch in dataset.take(1):
            for i in range(1):
                print(f'Image pixels: {features_batch["image_input"]}')
                print(f'Sentence embeddings: {features_batch["text_embedding"]}')
                print(f'Caption: {features_batch["caption"].numpy()}')
                label = label_batch.numpy()[i]
                print(f'Label : {label}')
        print("-----------------------------------------")



# Main class for the Image-Text Matching (ITM) task

# Import necessary libraries
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

class SVMClassifier:
    def __init__(self):
        # Initialize SVM classifier
        self.svm_classifier = SVC(kernel='linear', probability=True)

    def train(self, train_features, train_labels):
        # Flatten image features for SVM input
        train_features_flattened = [np.ravel(img_features.numpy()) for img_features, _ in train_features]

        # Convert labels to 1D array
        train_labels_flat = np.argmax(np.array([label.numpy() for _, label in train_labels]), axis=1)

        # Train SVM classifier
        self.svm_classifier.fit(train_features_flattened, train_labels_flat)

    def predict(self, features):
        # Flatten image features for SVM input
        features_flattened = [np.ravel(img_features.numpy()) for img_features, _ in features]

        # Predict using SVM classifier
        predictions = self.svm_classifier.predict(features_flattened)

        return predictions

# Update ITM_Classifier class to include SVM
class ITM_Classifier(ITM_DataLoader):
    def __init__(self):
        super().__init__()
        self.build_classifier_model()
        self.train_classifier_model()
        self.test_classifier_model()
        self.svm_classifier = SVMClassifier()
        
    # put together the feature representations above to create the image-text (multimodal) deep learning model
    def build_classifier_model(self):
        print(f'BUILDING model')
        img_input, vision_net = self.create_vision_encoder(num_projection_layers=1, projection_dims=128, dropout_rate=0.1)
        text_input, text_net = self.create_text_encoder(num_projection_layers=1, projection_dims=128, dropout_rate=0.1)
        net = tf.keras.layers.Concatenate(axis=1)([vision_net, text_net])
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(self.num_classes, activation='softmax', name=self.classifier_model_name)(net)
        self.classifier_model = tf.keras.Model(inputs=[img_input, text_input], outputs=net)
        self.classifier_model.summary()
        
    def train_classifier_model(self):
        super().train_classifier_model()
        # Train SVM classifier
        train_features = [(features['image_input'], label) for features, label in self.train_ds]
        self.svm_classifier.train(train_features)

    def test_classifier_model(self):
        super().test_classifier_model()
        # Test SVM classifier
        test_features = [(features['image_input'], label) for features, label in self.test_ds]
        svm_predictions = self.svm_classifier.predict(test_features)

        # Convert SVM predictions to match the format of NN predictions
        svm_predictions_nn_format = [[1, 0] if pred == 0 else [0, 1] for pred in svm_predictions]

        # Evaluate SVM performance
        svm_accuracy = accuracy_score(np.argmax([label.numpy() for _, label in self.test_ds], axis=1), svm_predictions)
        print("SVM Test Accuracy:", svm_accuracy)
        print("SVM Classification Report:")
        print(classification_report(np.argmax([label.numpy() for _, label in self.test_ds], axis=1), svm_predictions))


def plot_training_history(history):
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


# Let's create an instance of the main class

itm = ITM_Classifier()

plot_training_history(itm.history)
