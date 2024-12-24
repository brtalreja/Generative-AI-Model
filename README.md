# Generative AI Model using Python

## Overview
Generative AI models have emerged as powerful tools for a wide range of applications such as content creation, chatbots, automated story writing, and more. This project demonstrates the implementation of a text generation model trained on the Tiny Shakespeare dataset.

## Model Overview and Details
This project uses a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers to predict the next characters in a sequence, allowing it to generate coherent text. The model is built and trained using TensorFlow and the Tiny Shakespeare dataset, which contains excerpts of Shakespeare's works.

### Key Features
1. **Embedding Layer**: Converts input characters into dense vector representations.
2. **LSTM Layer**: Captures sequential dependencies and patterns in the text.
3. **Dense Layer**: Produces logits corresponding to the probability of each character in the vocabulary.

## Model Architecture
The model architecture consists of:
- **Embedding Layer**: Maps each character in the vocabulary to a 256-dimensional vector.
- **LSTM Layer**: A stateful LSTM with 1024 units to handle sequence data efficiently.
- **Dense Layer**: Outputs logits for each character in the vocabulary.

### Training Configuration
- **Loss Function**: Sparse categorical cross-entropy to compare logits and actual labels.
- **Optimizer**: Adam optimizer for efficient training.
- **Epochs**: 10 epochs with checkpoints saved after each epoch.

## Dataset
The Tiny Shakespeare dataset was used, which contains excerpts of Shakespeare’s works. The data was preprocessed to map each character to a numerical representation and split into sequences of length 100 for training.

## Results
After training, the model can generate Shakespeare-like text based on an input prompt. Below is an example:

```
Input: "QUEEN: So, let’s end this"
Output: "QUEEN: So, let's end this and make a great confession.
        KING: Thou art the mirror of virtue and valor..."
```

## How to Run the Project
1. **Clone the Repository**: Download or clone this project repository.
2. **Install Dependencies**: Ensure TensorFlow and TensorFlow Datasets are installed.
3. **Train the Model**: Run the script to train the model. Checkpoints will be saved after each epoch.
4. **Generate Text**: Use the `generate_text` function with a starting string to produce new text.

## Conclusion
This project builds on the foundational work provided in The Clever Programmer's Text Generation Model using Python by expanding the potential of LSTMs for generating coherent text sequences. While the model captures the essence of the dataset, I am working on further improvements that can be achieved with fine-tuning and other architectures.

## References
1. [LSTM](https://medium.com/@rebeen.jaff/what-is-lstm-introduction-to-long-short-term-memory-66bd3855b9ce)
2. [RNN and LSTM](https://aditi-mittal.medium.com/understanding-rnn-and-lstm-f7cdf6dfc14e)
3. [Project Reference](https://thecleverprogrammer.com/2024/01/22/text-generation-model-using-python/)
4. [Dataset](https://huggingface.co/datasets/karpathy/tiny_shakespeare)
5. [Text Generation](https://huggingface.co/tasks/text-generation)
