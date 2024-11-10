# Transformers
## Transformer family
- [BERT](bert/README.md)
- [GPT2](gpt/README.md)
- ALBERT
- ROBERT

## Transformers
- **Architecture**: Introduced by Vaswani et al. in the paper "Attention Is All You Need" (2017).
- **Revolutionized NLP**: Transformed natural language processing by moving away from recurrent neural networks.
- **Key Mechanism**: Utilizes self-attention mechanisms to process entire sequences in parallel.
- **Components**: Consists of an encoder-decoder structure, with the encoder processing input data and the decoder generating output sequences.
- **Scalability**: Highly scalable and efficient for training large models on massive datasets.

### Self-Attention
- **Core Concept**: Enables the model to weigh the importance of different words in a sentence, focusing on relevant parts of the input sequence.
- **Calculation**: Computes a set of `attention scores for each word, which determines its relevance to other words in the sequence`.
- **Benefits**: Captures dependencies regardless of distance between words, enhancing context understanding and improving performance on tasks like translation and summarization.
- **Usage**:
  - **Encoding**: Helps the encoder understand the context by attending to different parts of the input sequence simultaneously.
  - **Decoding**: Allows the decoder to dynamically focus on relevant parts of the encoded input when generating each word in the output sequence.

Transformers and self-attention have greatly improved the efficiency and effectiveness of NLP models, leading to significant advancements in tasks such as machine translation, text summarization, and language generation.

## Self attention vs attention 
- Self-attention in Transformers is different from the traditional attention mechanism used in encoder-decoder architectures. 
- Here are the key differences

### Traditional Attention in Encoder-Decoder
- **Context Vector**: In the traditional encoder-decoder model, the attention mechanism computes a context vector by weighing the importance of each encoder hidden state to the current decoder state.
- **Dependency**: The decoder relies on the encoder's hidden states to generate the context vector, and attention is applied only at the decoding stage.
- **Computation**: Attention scores are calculated for each encoder hidden state relative to the current decoder state, using a mechanism such as dot-product, additive, or scaled dot-product attention.

### Self-Attention in Transformers
- **Self-Contained Mechanism**: Self-attention operates within both the encoder and the decoder, allowing each word to attend to all other words in the same sequence, independent of position.
- **Positional Encoding**: Transformers include positional encodings to account for the order of words in the sequence, since self-attention itself is position-agnostic.
- **Parallel Processing**: Self-attention allows parallel processing of all words in the sequence, which significantly improves efficiency and speed compared to the sequential processing of traditional RNNs.
- **Layers**: Multiple layers of self-attention are stacked in the Transformer architecture, with each layer refining the representations learned in previous layers.

### Visual Comparison
- **Traditional Attention**: Focuses on the relationship between encoder states and decoder states, computing a context vector for each decoding step.
- **Self-Attention**: Computes relationships between all words in a sequence simultaneously, allowing each word to be represented with respect to the entire sequence context.

### Summary
- **Traditional Attention**: Applies at the decoding stage, computes a context vector using encoder states.
- **Self-Attention**: Applies within each layer of the encoder and decoder, enables parallel processing, and considers the entire sequence context simultaneously.

- These differences make self-attention in Transformers more powerful and efficient for `handling complex dependencies in sequences`, leading to significant advancements in NLP tasks.
