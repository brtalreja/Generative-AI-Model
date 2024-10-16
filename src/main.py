# Import required libraries
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

dataset, info = tfds.load('tiny_shakespeare', with_info = True, as_supervised=False)

# Getting all the texts from the dataset
text = next(iter(dataset['train']))['text'].numpy().decode('utf-8')

# Creating a vocab mapping from unique characters to indices
vocab = sorted(set(text))
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = np.array(vocab)

# Representing the characters numerically
text_as_int = np.array([char2idx[c] for c in text])

