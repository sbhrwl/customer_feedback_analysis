# LSTM
-  [Why LSTM?](#why-lstm)

## Why LSTM
### Problems in training simple RNNs
- Unstable Gradients
  - When we use **Unsaturated** activation function **RELU**
    - Update weight at t=0, if we are in +ve region, we would **Increase** the weight
      - This will result in increase in **Output**
    - Update weight at t=`, if we are in +ve region, we would **Increase** the weight
      - This will result in **FURTHER** increase in **Output**
    - This would eventually result in **EXPLODING GRADIENT**
      <img src="images/rnn-explodingGradient.png">

- **Solution:** 
    - Use **Saturated** activation function **tanh**
    - Perform **Gradient clipping**
    - Keep small **Learning rate**
     <img src="images/rnn-solution.png" width=500> 

## Bais initialisation
- Weights are initialised as 1 (instead of 0 in ANNs and other ML models)
- Initialising BIAS with 0 for RNN will result in totally discarding values
- for RNN Initialise `BIAS` with 1, because in LSTM we think about memory, so we would want to remeber initial values as well
