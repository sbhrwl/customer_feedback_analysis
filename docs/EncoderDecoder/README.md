# Encoder Decoder

## [Seq to Seq](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
### [Keras](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)
#### Inferencing
<img src="inferencing.png">

#### Examples
- [lstm-seq2seq](https://github.com/bond005/seq2seq)
- [Basic Encoder Decoder](https://colab.research.google.com/github/kmkarakaya/ML_tutorials/blob/master/seq2seq_Part_C_Basic_Encoder_Decoder.ipynb)
## 
- Encoder-decoder architectures were introduced to address several limitations and enhance the capabilities of sequence-to-sequence models, even when using advanced techniques like bidirectional RNNs. 
- Let's break down the key reasons:

### Handling Variable-Length Input and Output Sequences
- Bidirectional RNNs process sequences in both directions but are still designed to output sequences of the same length as the input. 
- Encoder-decoder architectures can handle variable-length input and output sequences, which is crucial for tasks like machine translation, where the source and target sentences may have different lengths.

### Improving Context Representation
- While bidirectional RNNs can capture information from both directions, they may still struggle with very long sequences. 
- An encoder-decoder architecture separates the task into two phases:
  - **Encoder**: Reads the entire input sequence and compresses it into a fixed-length context vector.
  - **Decoder**: Takes the context vector and generates the output sequence step-by-step. 
- This separation helps in **managing `long-term dependencies` more effectively**.

### **Providing Flexibility in Sequence Generation**
- The encoder-decoder framework offers flexibility in generating sequences. 
- The decoder can attend to different parts of the input sequence at each step, especially when combined with attention mechanisms. 
- This allows the model to focus on relevant parts of the input, improving the quality of the generated output.

### **Facilitating Attention Mechanisms**
- Attention mechanisms enhance the encoder-decoder architecture by **allowing the `decoder` to access different parts of the `encoder's output` dynamically**. 
- This is particularly useful in tasks like translation, where certain words in the input sequence need more attention when generating the corresponding output.

### **Example: Machine Translation**
- Consider translating a sentence from English to French:
- **Bidirectional RNN**: 
  - Processes the English sentence in both directions but may have limitations with very long sentences.
- **Encoder-Decoder**: 
  - The encoder processes the entire English sentence and creates a context vector. 
  - The decoder then generates the French sentence one word at a time, attending to different parts of the context vector as needed.

### **Illustration of Encoder-Decoder Architecture**
1. **Encoder**: 
   - Input: "The cat sits on the mat"
   - Output: Context vector (compressed representation of the input)

2. **Decoder**:
   - Input: Context vector
   - Output: "Le chat est assis sur le tapis" (translated sentence)

### **Code Example in TensorFlow/Keras:**

#### Encoder:
```python
from tensorflow.keras.layers import Input, LSTM, Embedding
from tensorflow.keras.models import Model

encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=vocab_size, output_dim=128)(encoder_inputs)
encoder_lstm = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]
```

#### Decoder:
```python
from tensorflow.keras.layers import Dense

decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(input_dim=vocab_size, output_dim=128)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
```

#### Full Model:
```python
from tensorflow.keras.models import Model

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, epochs=10)
```

### **Conclusion**
- The encoder-decoder architecture with attention mechanisms represents a significant advancement over bidirectional RNNs by addressing their limitations and enhancing sequence-to-sequence learning capabilities. 
- This architecture allows for better handling of variable-length sequences, improved long-term dependency management, and greater flexibility in sequence generation.