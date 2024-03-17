#####################################################
# VisionTransformerClassifier_Finetunned.py
# 
# Program extended from https://keras.io/examples/vision/bit/
#
# This program classifies images of flowers using a pre-trained Transformer known as
# BigTransfer (BiT). Since this is a self-contained program, method "read_image_data"
# may be found in other programs and method "run_experiment" could be standardised.
#
# Version: 1.0 -- more compact than original version, but see link above for details.
# Date: 06 March 2024
# Contact: hcuayahuitl@lincoln.ac.uk
#####################################################

import numpy as np
import pathlib
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
from tensorflow import keras

tf.get_logger().setLevel('ERROR')

def read_image_data(data_path, verbose=False):
    count=1
    X = []
    Y = []
    data_dir = pathlib.Path(data_path)
    print("Reading folder="+str(data_dir))
    dataset = tf.keras.utils.image_dataset_from_directory(data_dir)
    for image_batch, labels_batch in dataset:
        labels = labels_batch.numpy()
        if (verbose): print("Batch labels -> "+str(labels))
        for i in range(0, len(image_batch)):
            image = image_batch[i]
            label = labels[i]
            X.append(image)
            Y.append([label])
            if (verbose): print("["+str(count)+"] image="+str(image.shape)+" label="+str(label))
            count += 1
      
    X = np.array(X)
    Y = np.array(Y)
    class_names = dataset.class_names
    
    print("X="+str(X.shape))
    print("Y="+str(Y.shape))
    print("class_names="+str(class_names))
    
    return X, Y, class_names

class BiTModel(keras.Model):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.head = keras.layers.Dense(num_classes, kernel_initializer="zeros")
        self.bit_model = hub.load("https://tfhub.dev/google/bit/m-r50x1/1")

    def call(self, images):
        bit_embedding = self.bit_model(images)
        return self.head(bit_embedding)

def run_experiment(train_dir, test_dir, BATCH_SIZE, EPOCHS):
    x_train, y_train, class_names = read_image_data(train_dir)
    x_test, y_test, class_names = read_image_data(test_dir)
    print("x_train shape: %s y_train shape: %s" %(x_train.shape, y_train.shape))
    print("x_test shape: %s y_test shape: %s" % (x_test.shape, y_test.shape))

    optimizer = tfa.optimizers.AdamW(learning_rate=3e-5, weight_decay=0.0001)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    print("CREATING classifier model...")
    model = BiTModel(len(class_names))

    print("TRAINING & EVALUATING classifier model...")
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)
    _, accuracy = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    print("Test Accuracy: %s" % (round(accuracy * 100, 2)))


datapath_train = "datasets4transformers/flower_photos-resised_train"
datapath_test = "datasets4transformers/flower_photos-resised_test"
EPOCHS = 10
BATCH_SIZE = 32
run_experiment(datapath_train, datapath_test, BATCH_SIZE, EPOCHS)
