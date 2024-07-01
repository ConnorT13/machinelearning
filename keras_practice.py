import numpy as np
from tensorflow import keras

# Load the dataset (downloads automatically the first time)
(train_images, train_labels), _ = keras.datasets.mnist.load_data()

print(train_images.shape) # (60000, 28, 28)
print(train_labels.shape) # (60000,)
