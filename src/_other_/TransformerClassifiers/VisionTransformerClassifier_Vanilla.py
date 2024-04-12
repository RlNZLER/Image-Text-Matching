#####################################################
# VisionTransformerClassifier_Vanilla.py
# 
# Program extended and combined from the following two sources:
# https://keras.io/examples/vision/image_classification_with_vision_transformer/
# https://keras.io/examples/vision/vit_small_ds/
#
# This program classifies images of flowers using a Transformer trained from scratch,
# which focuses on the vanilla Transformer (ViT) instead of ViT for small-size data.
# Since this is a self-contained program, method "read_image_data" may be found in 
# other similar programs and method "run_experiment" is also similar to others.
#
# Version: 1.0 -- more compact than original version, but see link above for details.
# Date: 06 March 2024
# Contact: hcuayahuitl@lincoln.ac.uk
#####################################################

import numpy as np
import pathlib
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

tf.get_logger().setLevel('ERROR')

INPUT_SHAPE = (256,256,3)
IMAGE_SIZE = 72
PATCH_SIZE = 9
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

TRANSFORMER_LAYERS = 4
NUM_HEADS = 8
PROJECTION_DIM = 64
TRANSFORMER_UNITS = [PROJECTION_DIM * 2, PROJECTION_DIM,]
MLP_HEAD_UNITS = [256, 128]
LAYER_NORM_EPS = 1e-6

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

class Patches(keras.layers.Layer):
    def __init__(self, patch_size=PATCH_SIZE):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(keras.layers.Layer):
    def __init__(self, num_patches=NUM_PATCHES, projection_dim=PROJECTION_DIM):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = keras.layers.Dense(units=projection_dim)
        self.position_embedding = keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

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

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    return x

def create_classifier_model(NUM_CLASSES, data_augmentation):
    inputs = keras.layers.Input(shape=INPUT_SHAPE)
    augmented = data_augmentation(inputs)
    patches = Patches()(augmented)
    encoded_patches = PatchEncoder()(patches)

    for _ in range(TRANSFORMER_LAYERS):
        x1 = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = keras.layers.MultiHeadAttention(
                num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
            )(x1, x1)
        x2 = keras.layers.Add()([attention_output, encoded_patches])
        x3 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=TRANSFORMER_UNITS, dropout_rate=0.1)
        encoded_patches = keras.layers.Add()([x3, x2])

    representation = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = keras.layers.Flatten()(representation)
    representation = keras.layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=MLP_HEAD_UNITS, dropout_rate=0.5)
    logits = keras.layers.Dense(NUM_CLASSES)(features)
    model = keras.Model(inputs=inputs, outputs=logits)
    model.summary()
    return model

def run_experiment(train_dir, test_dir, BATCH_SIZE, EPOCHS):
    x_train, y_train, class_names = read_image_data(train_dir)
    x_test, y_test, class_names = read_image_data(test_dir)
    print("x_train shape: %s y_train shape: %s" %(x_train.shape, y_train.shape))
    print("x_test shape: %s y_test shape: %s" % (x_test.shape, y_test.shape))

    optimizer = tfa.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    print("AUGMENTING DATA...")
    data_augmentation = keras.Sequential(
        [
            keras.layers.Normalization(),
            keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(factor=0.02),
            keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ],
        name="data_augmentation",
        )
    data_augmentation.layers[0].adapt(x_train)

    print("CREATING classifier model...")
    model = create_classifier_model(len(class_names), data_augmentation)

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
