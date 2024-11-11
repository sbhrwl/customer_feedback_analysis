# Transformers
## Transformer family
- [Self Attention](#self-attention)
- [Introduction](#introduction)
- [Building block of transformers](#building-block-of-transformers)
  - [Feed forward neural network](#feed-forward-neural-network)
- [BERT](bert/README.md)
- [GPT2](gpt/README.md)
- ALBERT
- ROBERT

### Self Attention
- **Core Concept**: Enables the model to weigh the importance of different words in a sentence, focusing on relevant parts of the input sequence.
- **Calculation**: Computes a set of `attention scores for each word, which determines its relevance to other words in the sequence`.
- **Benefits**: Captures dependencies regardless of distance between words, enhancing context understanding and improving performance on tasks like translation and summarization.
- **Usage**:
  - **Encoding**: Helps the encoder understand the context by attending to different parts of the input sequence simultaneously.
  - **Decoding**: Allows the decoder to dynamically focus on relevant parts of the encoded input when generating each word in the output sequence.

## Introduction
- **Architecture**: Introduced by Vaswani et al. in the paper "Attention Is All You Need" (2017).
- **Revolutionized NLP**: Transformed natural language processing by moving away from recurrent neural networks.
- **Key Mechanism**: Utilizes self-attention mechanisms to process entire sequences in parallel.
- **Components**: Consists of an encoder-decoder structure, with the encoder processing input data and the decoder generating output sequences.
- **Scalability**: Highly scalable and efficient for training large models on massive datasets.


## Building block of transformers
- The fundamental building block of transformers is the **transformer layer**, which leverages self-attention mechanisms and feed-forward neural networks.
- Here’s a breakdown of its core components

### Self-Attention Mechanism
- **Purpose**
  - Allows the model to weigh the importance of different words in a sequence.
- **Components**:
  - **Query (Q)**: A representation of the current word.
  - **Key (K)**: A representation used to calculate the attention score.
  - **Value (V)**: A representation that is combined based on the attention scores.
- **Process**
  - Computes attention scores using Q and K, then combines V based on these scores.

### Multi-Head Attention
- **Purpose**
  - Enhances the model’s ability to focus on different parts of the sequence simultaneously.
- **Components**: Multiple self-attention heads run in parallel.
- **Process**
  - Each head performs self-attention independently, and their outputs are concatenated and linearly transformed.

### Feed-Forward Neural Network
- **Purpose**
  - Adds non-linearity and helps in capturing complex patterns.
- **Components**:
  - **Dense Layers**: Fully connected layers with activation functions.
- **Process**
  - Applies two linear transformations with a ReLU activation in between.

### Layer Normalization and Residual Connections
- **Purpose**
  - Stabilizes training and helps in retaining information across layers.
- **Components**: 
  - **Layer Norm**: Normalizes inputs to the layer.
  - **Residual Connections**: Adds input to the output of the layer.
- **Process**
  - Applies normalization and adds residual connections before and after each sub-layer (self-attention and feed-forward).

### Positional Encoding
- **Purpose**
  - Provides information about the position of words in the sequence.
- **Components**: Sinusoidal functions that encode position information.
- **Process**
  - Adds positional encodings to the input embeddings to retain positional information.

### Visualization of a Transformer Layer
- These components are stacked to build the encoder and decoder in transformers, forming a powerful model capable of handling complex sequence tasks efficiently.
```
+-------------------+
|   Input Embedding |
| + Positional Enc. |
+-------------------+
         |
+-------------------+
| Multi-Head        |
| Self-Attention    |
+-------------------+
         |
+-------------------+
| Add & Norm        |
+-------------------+
         |
+-------------------+
| Feed-Forward      |
| Neural Network    |
+-------------------+
         |
+-------------------+
| Add & Norm        |
+-------------------+
         |
      Output
```
### Feed forward neural network
- In the context of transformer models, the feed-forward neural network (FFN) is a crucial component that adds non-linearity and depth to the model. Here’s what it typically includes:

#### Two Linear (Dense) Layers
- **First Linear Layer**: Projects the input embeddings to a higher-dimensional space.
  - **Input Dimension**: \( d_{model} \)
  - **Output Dimension**: \( d_{ff} \) (often larger than \( d_{model} \))
- **Second Linear Layer**: Projects the higher-dimensional space back to the original dimension.
  - **Input Dimension**: \( d_{ff} \)
  - **Output Dimension**: \( d_{model} \)

#### Activation Function
- **Purpose**: Introduces non-linearity into the model.
- **Common Activation**: ReLU (Rectified Linear Unit)

#### Dropout (Optional but Common)
- **Purpose**: Prevents overfitting by randomly setting a fraction of the output units to zero during training.
- **Typical Dropout Rate**: Around 0.1 or 10%
#### Process Flow
1. **Input**: The output from the self-attention layer.
2. **First Linear Transformation**: Applies the first dense layer.
3. **Activation**: Applies the ReLU activation function.
4. **Second Linear Transformation**: Applies the second dense layer.
5. **Dropout (if used)**: Applies dropout to the output.
6. **Output**: The final transformed output.

#### Illustration
```plaintext
Input Embedding
      │
First Linear Layer
      │
     ReLU
      │
Second Linear Layer
      │
   Dropout
      │
   Output
```

#### Summary of feed forward neural network
- **First Linear Layer**: Expands the dimensionality.
- **ReLU Activation**: Adds non-linearity.
- **Second Linear Layer**: Reduces the dimensionality back to the original size.
- **Dropout**: Helps regularize the model.
- This feed-forward network is applied independently to each position in the sequence and the same parameters are used for each position.
- It's a key component that enhances the model's capacity to capture complex patterns and relationships in the data.

### Summary
- **Self-Attention Mechanism**: Focuses on different parts of the sequence.
- **Multi-Head Attention**: Allows multiple attentions in parallel.
- **Feed-Forward Network**: Adds complexity to the model.
- **Layer Normalization & Residual Connections**: Improve stability and information flow.
- **Positional Encoding**: Retains word positions in the sequence.

