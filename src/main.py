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

# create training examples and targets
seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)

# create training sequences
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

# Function to split the input and target text
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)