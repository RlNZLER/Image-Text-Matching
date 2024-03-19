import tensorflow as tf

# Print out the available physical devices
print("Available Physical Devices:")
physical_devices = tf.config.list_physical_devices()
for device in physical_devices:
    print(device)

# Print out the available GPUs
print("\nAvailable GPUs:")
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print(gpu)

# Print out information about GPU usage by TensorFlow
print("\nTensorFlow GPU Usage:")
print(tf.config.experimental.list_logical_devices('GPU'))


import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)