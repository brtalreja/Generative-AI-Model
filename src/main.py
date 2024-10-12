# Import required libraries
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

dataset, info = tfds.load('tiny_shakespeare', with_info = True, as_supervised=False)
