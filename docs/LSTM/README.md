# Long Short Term Memory
- [Why LSTM?](#why-lstm)
  - [Challenges with Traditional RNNs](#challenges-with-traditional-rnns)
- [LSTM](#lstm)
  - [Blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  - [Keras Implementation](#keras-implementation)
  - [Components](#components)
    - [States](#states)
    - [Output of FC layers](#output-of-fc-layers)
    - [Gates](#gates)
  - [LSTM in action](LSTMInAction.md)
- [Bais initialisation](#bais-initialisation)

## Why LSTM
### Problems in training simple RNNs
#### Problem 1: Unstable Gradients
- When we use **Unsaturated** activation function **RELU**
  - Update weight at t=0, if we are in +ve region, we would **Increase** the weight
    - This will result in increase in **Output**
  - Update weight at t=1, if we are in +ve region, we would **Increase** the weight
    - This will result in **FURTHER** increase in **Output**
  - This would eventually result in **EXPLODING GRADIENT**
    <img src="images/rnn-explodingGradient.png">

#### Solution
- Use **Saturated** activation function **tanh**
- Perform **Gradient clipping**
  - Specify in which range we want tanh function to vary (**Fix the region**)
  - If we are in certain range on tanh function, we will get max value (**-5 to +5**)
  - **Gradient clipping** to prevent **Vanishing Gradient** problem
- Keep small **Learning rate** when using RELU, 
  - This will not allow large weight update hence preventing exploding gradient problem
 <img src="images/rnn-solution.png" width=500>

#### Problem 2: Low memory retention
<img src="images/low_memory_retention.png" width=500>

#### Solution
- LSTM, Long Short Term Memory and
- GRU, Gated Recurrent Unit

## LSTM
- Paper: In 1997 by Sepp Hochreiter
- LSTM architecture helps to remember the important things for **Long** term duration
<img src="images/lstm.png" width=500>

### Keras Implementation
- Option 1: Optimized for GPU
```python
tf.keras.layers.LSTM(20,)
```
- Option 2: Offers more customization options but not optimised for GPUs
```python
tf.keras.RNN(tf.keras.layers.LSTMCells(20))
```
- Option 3: Time Distributed
```python
tf.keras.TimeDistributed(tf.keras.layers.Dense(20))
```
### Components
#### States
- $c_{t}$ : **Long** term state
- $h_{t}$ : **Short** term state
- $c_{t-1}$ : **Previous Long** term state
- $h_{t-1}$ : **Previous Short** term state

#### Output of FC layers
- $f_{t}$ : has **Sigmoid** activation 
- $g_{t}$ : has **tanh** activation
- $i_{t}$ : has **Sigmoid** activation
- $o_{t}$ : has **Sigmoid** activation
- All FC layers will take 2 inputs
  - $x_{t}$ : **Current Input feature** 
  - $h_{t-1}$ : **Previous Short** term state
- $y_{t}$ : **Current Output** 
#### Gates
- Gates would generate the memory elements
- **Forget** Gate
- **Input** Gate
  - **Input** Gate adds to **Forget** Gate to produce **Long** term state
- **Output** Gate

##### Forget Gate
- Takes 2 Inputs
  - $c_{t-1}$ : **Previous Long** term state
  - $f_{t}$ : with **Sigmoid** activation
- Produces an Output that results in $c_{t}$ : **Long** term state
##### Input Gate
- Takes 2 Inputs
  - $g_{t}$ : with **tanh** activation
  - $i_{t}$ : with **Sigmoid** activation
- The output will **ADD** to Forget gate Output that results in $c_{t}$ : **Long** term state
##### Output Gate
- $c_{t}$ : **Long** term state, passed through **tanh**
- $o_{t}$ : with **Sigmoid** activation
- Produces an Output that results in $h_{t}$ : **Short** term state

## Bais initialisation
- Weights are initialised as 1 (instead of 0 in ANNs and other ML models)
- Initialising BIAS with 0 for RNN will result in totally discarding values
- for RNN Initialise `BIAS` with 1, because in LSTM we think about memory, so we would want to remeber initial values as well

## Challenges with Traditional RNNs
1. **Vanishing Gradient Problem**:
   - **Issue**: During training, the gradients used to update the network's weights can become very small, leading to minimal changes in weights and causing the network to learn very slowly or not at all for long-term dependencies.
   - **Impact**: This makes it difficult for traditional RNNs to learn relationships between data points that are far apart in the sequence.
2. **Short-Term Memory**:
   - **Issue**: RNNs tend to forget information as the length of the sequence increases.
   - **Impact**: They struggle with tasks that require understanding long-term dependencies or context, such as language translation or text generation.

## **How LSTMs Address These Challenges:**
1. **Memory Cells**:
   - LSTMs have a unique architecture that includes memory cells, which can store information for long periods. These cells can read, write, and delete information, allowing the network to retain important information over long sequences.
2. **Gates Mechanism**:
   - LSTMs use three gates (input, output, and forget gates) to control the flow of information. These gates decide which information to keep, which to discard, and what new information to add, effectively managing long-term dependencies.
3. **Avoiding Vanishing Gradient**:
   - By using these gates and memory cells, LSTMs mitigate the vanishing gradient problem, enabling the network to learn long-term dependencies more effectively.

### **Example Use Cases**:
1. **Language Translation**:
   - LSTMs can understand and remember long-term dependencies in sentences, improving the accuracy of translations.
2. **Speech Recognition**:
   - They can capture long-term patterns in audio signals, leading to better recognition performance.
3. **Time Series Prediction**:
   - LSTMs can handle long-term trends and patterns in time series data, making them effective for forecasting.

### **Conclusion**:
LSTMs are a significant advancement over traditional RNNs, allowing for more effective handling of long-term dependencies and overcoming the limitations of vanishing gradients. This makes them well-suited for complex tasks requiring memory of past information over long sequences.