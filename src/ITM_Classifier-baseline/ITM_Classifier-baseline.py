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

import os
import csv
import pickle
import random
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from official.nlp import optimization
import matplotlib.pyplot as plt


# Class for loading image and text data


class ITM_DataLoader:
    BATCH_SIZE = 16
    IMAGE_SIZE = (224, 224)
    IMAGE_SHAPE = (224, 224, 3)
    SENTENCE_EMBEDDING_SHAPE = 384
    AUTOTUNE = tf.data.AUTOTUNE
    DATA_PATH = "D:\_GITHUB_\Image-Text-Matching\data"
    IMAGES_PATH = DATA_PATH + "/images"
    train_data_file = DATA_PATH + "/flickr8k.TrainImages.txt"
    dev_data_file = DATA_PATH + "/flickr8k.DevImages.txt"
    test_data_file = DATA_PATH + "/flickr8k.TestImages.txt"
    sentence_embeddings_file = DATA_PATH + "/flickr8k.cmp9137.sentence_transformers.pkl"
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
        with open(self.sentence_embeddings_file, "rb") as f:
            data = pickle.load(f)
            for sentence, dense_vector in data.items():
                # print("*sentence=",sentence)
                sentence_embeddings[sentence] = dense_vector
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
        print("LOADING data from " + str(data_files))
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
                # print("I=%s T=%s _L=%s L=%s" % (img_name, text, raw_label, label))

                # get sentence embeddings (of textual captions)
                text_sentence_embedding = self.sentence_embeddings[text]
                text_sentence_embedding = tf.constant(text_sentence_embedding)

                image_data.append(img_name)
                embeddings_data.append(text_sentence_embedding)
                text_data.append(text)
                label_data.append(label)

        print("|image_data|=" + str(len(image_data)))
        print("|text_data|=" + str(len(text_data)))
        print("|label_data|=" + str(len(label_data)))

        # prepare a tensorflow dataset using the lists generated above
        dataset = tf.data.Dataset.from_tensor_slices(
            (image_data, embeddings_data, text_data, label_data)
        )
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
                print(f"Label : {label}")
        print("-----------------------------------------")


# Main class for the Image-Text Matching (ITM) task


class ITM_Classifier(ITM_DataLoader):
    epochs = 2
    learning_rate = 3e-5
    class_names = {"match", "no-match"}
    num_classes = len(class_names)
    classifier_model = None
    history = None
    classifier_model_name = "ITM_Classifier-baseline"
    training_time = None

    def __init__(self):
        super().__init__()
        self.build_classifier_model()
        self.train_classifier_model()
        self.test_classifier_model()

    # return learnt feature representations of input data (images)
    def create_vision_encoder(
        self, num_projection_layers, projection_dims, dropout_rate
    ):
        img_input = layers.Input(shape=self.IMAGE_SHAPE, name="image_input")
        cnn_layer = layers.Conv2D(16, 3, padding="same", activation="relu")(img_input)
        cnn_layer = layers.MaxPooling2D()(cnn_layer)
        cnn_layer = layers.Conv2D(32, 3, padding="same", activation="relu")(cnn_layer)
        cnn_layer = layers.MaxPooling2D()(cnn_layer)
        cnn_layer = layers.Conv2D(64, 3, padding="same", activation="relu")(cnn_layer)
        cnn_layer = layers.MaxPooling2D()(cnn_layer)
        cnn_layer = layers.Dropout(dropout_rate)(cnn_layer)
        cnn_layer = layers.Flatten()(cnn_layer)
        outputs = self.project_embeddings(
            cnn_layer, num_projection_layers, projection_dims, dropout_rate
        )
        return img_input, outputs

    # return learnt feature representations based on dense layers, dropout, and layer normalisation
    def project_embeddings(
        self, embeddings, num_projection_layers, projection_dims, dropout_rate
    ):
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
        text_input = keras.Input(
            shape=self.SENTENCE_EMBEDDING_SHAPE, name="text_embedding"
        )
        outputs = self.project_embeddings(
            text_input, num_projection_layers, projection_dims, dropout_rate
        )
        return text_input, outputs

    # put together the feature representations above to create the image-text (multimodal) deep learning model
    def build_classifier_model(self):
        print(f"BUILDING model")
        img_input, vision_net = self.create_vision_encoder(
            num_projection_layers=1, projection_dims=128, dropout_rate=0.1
        )
        text_input, text_net = self.create_text_encoder(
            num_projection_layers=1, projection_dims=128, dropout_rate=0.1
        )
        net = tf.keras.layers.Concatenate(axis=1)([vision_net, text_net])
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(
            self.num_classes, activation="softmax", name=self.classifier_model_name
        )(net)
        self.classifier_model = tf.keras.Model(
            inputs=[img_input, text_input], outputs=net
        )
        self.classifier_model.summary()

    def save_model(self):
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)  # Ensure directory exists
        model_path = os.path.join(model_dir, self.classifier_model_name)
        history_path = os.path.join(
            model_dir, f"{self.classifier_model_name}_history.pkl"
        )
        print("SAVING model history to", model_path)
        # self.classifier_model.save(model_path)  # Save the model
        with open(history_path, "wb") as f:
            pickle.dump(self.history.history, f)  # Save the training history

    def train_classifier_model(self):
        print(f"TRAINING model")
        start_time = datetime.datetime.now()
        steps_per_epoch = tf.data.experimental.cardinality(self.train_ds).numpy()
        num_train_steps = steps_per_epoch * self.epochs
        num_warmup_steps = int(0.2 * num_train_steps)

        loss = tf.keras.losses.KLDivergence()
        metrics = [
            BinaryAccuracy(name="binary_accuracy"),
            Precision(name="precision"),
            Recall(name="recall"),
        ]
        optimizer = optimization.create_optimizer(
            init_lr=self.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            optimizer_type="adamw",
        )

        self.classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # uncomment the next line if you wish to make use of early stopping during training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=11, restore_best_weights=True)
        ]

        self.history = self.classifier_model.fit(
            x=self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs,
            callbacks=callbacks,
        )
        self.save_model()
        end_time = datetime.datetime.now()
        self.training_time = end_time - start_time
        print("MODEL TRAINED!")
        print(f"Training completed in {self.training_time}")

    def test_classifier_model(self):
        print(
            "TESTING classifier model (showing a sample of image-text-matching predictions)..."
        )
        num_classifications = 0
        num_correct_predictions = 0

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
                predicted_class = (
                    "[1 0]" if probability_match > probability_nomatch else "[0 1]"
                )
                if str(groundtruth[batch_index]) == predicted_class:
                    num_correct_predictions += 1
                num_classifications += 1

                # print a sample of predictions -- about 10% of all possible
                if random.random() < 0.1:
                    caption = captions[batch_index]
                    file_name = file_names[batch_index].decode("utf-8")
                    print(
                        "ITM=%s PREDICTIONS: match=%s, no-match=%s \t -> \t %s"
                        % (caption, probability_match, probability_nomatch, file_name)
                    )

        # reveal test performance using our own calculations above
        accuracy = num_correct_predictions / num_classifications
        print("TEST accuracy=%4f" % (accuracy))

        # reveal test performance using Tensorflow calculations
        loss, accuracy, precision, recall = self.classifier_model.evaluate(self.test_ds)
        print(
            f"Tensorflow test method: LOSS: {loss}; ACCURACY: {accuracy}: PRECISION: {precision}; RECALL: {recall}"
        )


# Calculate F1 scores from precision and recall values.
def calculate_f1_score(precision, recall):
    if (precision + recall) == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


# Log final metrics to a CSV file for tracking and comparison.
def log_metrics(itm):
    # Define the CSV file path
    csv_file = os.path.join("logs", "itm_final_metrics_log.csv")
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    # Prepare the header for the CSV
    fieldnames = [
        "Timestamp",
        "Model Name",
        "Learning Rate",
        "Epoch",
        "Train Loss",
        "Train Accuracy",
        "Validation Loss",
        "Validation Accuracy",
        "Test Loss",
        "Test Accuracy",
        "Test Precision",
        "Test Recall",
        "Test F1 Score",
        "Training Time",
    ]

    # Check if the file exists to decide whether to add headers
    file_exists = os.path.exists(csv_file)

    # Open the CSV file for writing
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header only if the file does not exist
        if not file_exists:
            writer.writeheader()

        # Collect the last epoch training and validation metrics
        history = itm.history.history
        final_epoch = len(history["loss"]) - 1
        train_loss = history["loss"][final_epoch]
        train_accuracy = history["binary_accuracy"][final_epoch]
        val_loss = history["val_loss"][final_epoch]
        val_accuracy = history["val_binary_accuracy"][final_epoch]

        # Evaluate the model on the test dataset and get test metrics
        test_results = itm.classifier_model.evaluate(itm.test_ds)
        test_loss = test_results[0]
        test_accuracy = test_results[1]
        test_precision = test_results[2]
        test_recall = test_results[3]
        test_f1_score = calculate_f1_score(test_precision, test_recall)

        # Write metrics to the CSV
        writer.writerow(
            {
                "Timestamp": datetime.datetime.now().isoformat(),
                "Model Name": itm.classifier_model_name,
                "Learning Rate": itm.learning_rate,
                "Epoch": itm.epochs,
                "Train Loss": train_loss,
                "Train Accuracy": train_accuracy,
                "Validation Loss": val_loss,
                "Validation Accuracy": val_accuracy,
                "Test Loss": test_loss,
                "Test Accuracy": test_accuracy,
                "Test Precision": test_precision,
                "Test Recall": test_recall,
                "Test F1 Score": test_f1_score,
                "Training Time": itm.training_time.total_seconds(),  # Convert to seconds
            }
        )


# Plot training history metrics for accuracy, loss, precision, recall, and F1 score.
def plot_training_history(itm):
    # Extract metrics from history
    history_data = itm.history.history
    acc = history_data.get("binary_accuracy", [])
    val_acc = history_data.get("val_binary_accuracy", [])
    loss = history_data.get("loss", [])
    val_loss = history_data.get("val_loss", [])
    precision = history_data.get("precision", [])
    val_precision = history_data.get("val_precision", [])
    recall = history_data.get("recall", [])
    val_recall = history_data.get("val_recall", [])

    # Calculate F1 Scores for each epoch
    f1 = [calculate_f1_score(prec, rec) for prec, rec in zip(precision, recall)]
    val_f1 = [
        calculate_f1_score(prec, rec) for prec, rec in zip(val_precision, val_recall)
    ]

    plt.figure(figsize=(12, 8))

    # Subplot for Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Subplot for Loss
    plt.subplot(2, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Subplot for Precision and Recall
    plt.subplot(2, 2, 3)
    plt.plot(precision, label="Train Precision")
    plt.plot(val_precision, label="Validation Precision")
    plt.plot(recall, label="Train Recall")
    plt.plot(val_recall, label="Validation Recall")
    plt.title("Precision and Recall")
    plt.xlabel("Epochs")
    plt.ylabel("Values")
    plt.legend()

    # Subplot for F1 Score
    plt.subplot(2, 2, 4)
    plt.plot(f1, label="Train F1 Score")
    plt.plot(val_f1, label="Validation F1 Score")
    plt.title("F1 Score")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        f"./plots/{itm.classifier_model_name}_training_history_plots.png"
    )  # Save the figure
    plt.show()


# Set up GPU memory growth to avoid memory allocation issues.
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Initialize the ITM classifier and plot the training history.
itm = ITM_Classifier()
log_metrics(itm)
plot_training_history(itm)
