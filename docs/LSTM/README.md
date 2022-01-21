# Long Short Term Memory
-  [Why LSTM?](#why-lstm)
-  [Bais initialisation](#bais-initialisation)

## Why LSTM
### Problems in training simple RNNs
#### Problem 1: Unstable Gradients
- When we use **Unsaturated** activation function **RELU**
  - Update weight at t=0, if we are in +ve region, we would **Increase** the weight
    - This will result in increase in **Output**
  - Update weight at t=`, if we are in +ve region, we would **Increase** the weight
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
<img src="images/lstm.png" width=500>

## Bais initialisation
- Weights are initialised as 1 (instead of 0 in ANNs and other ML models)
- Initialising BIAS with 0 for RNN will result in totally discarding values
- for RNN Initialise `BIAS` with 1, because in LSTM we think about memory, so we would want to remeber initial values as well
