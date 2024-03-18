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

import io
import os
import sys
import csv
import time
import einops
import pickle
import random
import subprocess
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from official.nlp import optimization
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader  # Make sure to import ImageReader
from PIL import Image

# Verify TensorFlow can detect the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"Num GPUs Available: {len(gpus)}")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if len(gpus) > 0:
    print(f"Using GPU: {gpus[0].device_type} {gpus[0].name}")
else:
    print("No GPU detected. Running on CPU.")
    

def append_dict_to_csv(row_dict, csv_file_name):
    # Open the file in append mode (or create it if it doesn't exist)
    with open(csv_file_name, mode='a', newline='', encoding='utf-8') as csv_file:
        # Create a writer object
        writer = csv.DictWriter(csv_file, fieldnames=row_dict.keys())
        
        # Write the data row
        writer.writerow(row_dict)

def insert_image_in_pdf(image_path, pdf_canvas, x, y, width=224, height=224):
    """Insert an image into the PDF canvas."""
    img = Image.open(image_path)
    
    # Resize the image by 50%
    new_width = int(width * 0.5)
    new_height = int(height * 0.5)
    img = img.resize((new_width, new_height))
    
    # Convert PIL Image to a data stream compatible with reportlab
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    pdf_canvas.drawImage(ImageReader(img_buffer), x, y, new_width, new_height)

def generate_pdf_from_csv(csv_file_name, output_file_name):
    c = canvas.Canvas(output_file_name, pagesize=letter)
    width, height = letter
    
    # Define table headers
    headers = ["S/N", "Image", "Caption", "Match", "Un-match"]
    
    # Starting positions
    x = 50
    y = height - 50  # Start from the top of the page
    row_height = 150  # Adjusted for image height
    col_widths = [30, 180, 180, 80, 80]  # Adjust the column widths as needed
    
    # Draw Table Header
    for idx, header in enumerate(headers):
        c.drawString(x + sum(col_widths[:idx]), y, header)
    
    y -= row_height
    
    # Read data from CSV and fill in the table rows
    with open(csv_file_name, mode='r', newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for sn, row in enumerate(reader, start=1):
            c.drawString(x, y + (row_height - 12) / 2, str(sn))
            
            # Insert image
            insert_image_in_pdf(row["Image"], c, x + col_widths[0], y)
            
            # Text columns
            # Remove the "b'" characters from the start of the caption
            caption = row["Caption"].lstrip("b'")

            # Split caption into multiple lines if it's too long
            caption_lines = [caption[i:i+25] for i in range(0, len(caption), 25)]
            for idx, line in enumerate(caption_lines):
                c.drawString(x + sum(col_widths[:2]), y + (row_height - 12) / 2 - (idx * 12 * 0.8), line)
            
            c.drawString(x + sum(col_widths[:3]), y + (row_height - 12) / 2, row["Match"])
            c.drawString(x + sum(col_widths[:4]), y + (row_height - 12) / 2, row["Un-match"])
            
            y -= row_height
            if y < 100:  # Check for page end
                c.showPage()  # Start a new page
                y = height - 50  # Reset y position
                # Re-draw Table Header on new page
                for idx, header in enumerate(headers):
                    c.drawString(x + sum(col_widths[:idx]), y, header)
                y -= row_height
    
    c.save()
    subprocess.Popen(['xdg-open', output_file_name])
    
    
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

class ITM_Classifier(ITM_DataLoader):
    epochs = 10
    learning_rate = 3e-5
    class_names = {'match', 'no-match'}
    num_classes = len(class_names)
    classifier_model = None
    history = None
    classifier_model_name = 'ITM_Classifier-flickr'

    def __init__(self):
        super().__init__()
        self.build_classifier_model()
        self.train_classifier_model()
        self.test_classifier_model()

    # return learnt feature representations of input data (images)
    def create_vision_encoder(self, num_projection_layers, projection_dims, dropout_rate):
        img_input = layers.Input(shape=self.IMAGE_SHAPE, name="image_input")
        cnn_layer = layers.Conv2D(16, 3, padding='same', activation='relu')(img_input)
        cnn_layer = layers.MaxPooling2D()(cnn_layer)
        cnn_layer = layers.Conv2D(32, 3, padding='same', activation='relu')(cnn_layer)
        cnn_layer = layers.MaxPooling2D()(cnn_layer)
        cnn_layer = layers.Conv2D(64, 3, padding='same', activation='relu')(cnn_layer)
        cnn_layer = layers.MaxPooling2D()(cnn_layer)
        cnn_layer = layers.Dropout(dropout_rate)(cnn_layer)
        cnn_layer = layers.Flatten()(cnn_layer)
        outputs = self.project_embeddings(cnn_layer, num_projection_layers, projection_dims, dropout_rate)
        return img_input, outputs

    # return learnt feature representations based on dense layers, dropout, and layer normalisation
    def project_embeddings(self, embeddings, num_projection_layers, projection_dims, dropout_rate):
        projected_embeddings = layers.Dense(units=projection_dims)(embeddings)
        for _ in range(num_projection_layers):
            x = tf.nn.gelu(projected_embeddings)
            x = layers.Dense(projection_dims)(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Add()([projected_embeddings, x])
            projected_embeddings = layers.LayerNormalization()(x)
        return projected_embeddings

    # return learnt feature representations of input data (text embeddings in the form of dense vectors)
    def create_text_encoder(self, num_projection_layers, projection_dims, dropout_rate):
        text_input = keras.Input(shape=self.SENTENCE_EMBEDDING_SHAPE, name='text_embedding')
        outputs = self.project_embeddings(text_input, num_projection_layers, projection_dims, dropout_rate)
        return text_input, outputs

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
        print(f'TRAINING model')
        steps_per_epoch = tf.data.experimental.cardinality(self.train_ds).numpy()
        num_train_steps = steps_per_epoch * self.epochs
        num_warmup_steps = int(0.2*num_train_steps)

        loss = tf.keras.losses.KLDivergence()
        metrics = tf.keras.metrics.BinaryAccuracy()
        optimizer = optimization.create_optimizer(init_lr=self.learning_rate,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

        self.classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # uncomment the next line if you wish to make use of early stopping during training
        #callbacks = [tf.keras.callbacks.EarlyStopping(patience=11, restore_best_weights=True)]

        self.history = self.classifier_model.fit(x=self.train_ds, validation_data=self.val_ds, epochs=self.epochs)#, callbacks=callbacks)
        print("model trained!")

    def test_classifier_model(self):
        print("TESTING classifier model (showing a sample of image-text-matching predictions)...")
        num_classifications = 0
        num_correct_predictions = 0
        
        fieldnames = ["Image", "Caption", "Match", "Un-match"]
        filename = "output/Test_sample.csv"

        # Delete the existing file if it exists
        if os.path.exists(filename):
            os.remove(filename)

        with open(filename, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

        # read test data for ITM classification
        for features, groundtruth in self.test_ds:
            groundtruth = groundtruth.numpy()
            predictions = self.classifier_model(features)
            predictions = predictions.numpy()
            captions = features["caption"].numpy()
            file_names = features["file_name"].numpy()

            # read test data per batch
            for batch_index in range(0, len(groundtruth)):
                predicted_values = predictions[batch_index]
                probability_match = predicted_values[0]
                probability_nomatch = predicted_values[1]
                predicted_class = "[1 0]" if probability_match > probability_nomatch else "[0 1]"
                if str(groundtruth[batch_index]) == predicted_class: 
                    num_correct_predictions += 1
                num_classifications += 1

                # print a sample of predictions -- about 10% of all possible
                if random.random() < 0.1:
                    caption = captions[batch_index]
                    file_name = file_names[batch_index].decode("utf-8")
                    print("ITM=%s PREDICTIONS: match=%s, no-match=%s \t -> \t %s" % (caption, probability_match, probability_nomatch, file_name))
                    data = {
                                "file_name": file_name,
                                "caption": caption,
                                "match": probability_match,
                                "unmatch": probability_nomatch
                            }
                    append_dict_to_csv(data, "output/Test_sample.csv")
                    
        # reveal test performance using our own calculations above
        accuracy = num_correct_predictions/num_classifications
        print("TEST accuracy=%4f" % (accuracy))

        # reveal test performance using Tensorflow calculations
        loss, accuracy = self.classifier_model.evaluate(self.test_ds)
        print(f'Tensorflow test method: Loss: {loss}; ACCURACY: {accuracy}')


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
generate_pdf_from_csv("output/Test_sample.csv", "output/Test_sample.pdf")
