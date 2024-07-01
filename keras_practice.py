import numpy as np
from tensorflow import keras
import mnist

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

train_images = (train_images / 255) - 0.5
test_labels = (test_images / 255) - 0.5

# flattens images by taking the 28 * 28 2-D array, and transforms it into a 1-D Array
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1,784))

print(train_images.shape)
print(test_images.shape)