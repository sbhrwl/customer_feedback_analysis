# LSTM in action

## Consider different states coming in and going out from a LSTM cell
<img src="images/states_lstm.png">

## Long Term memory
### Forget gate
- **Previous Short** term state and **Current Input feature** will go through an **ANN**
- Result of ANN will go through **Sigmoid** function: <img src="https://render.githubusercontent.com/render/math?math=f_{t}">
- <img src="https://render.githubusercontent.com/render/math?math=f_{t}"> will now go through **MULTIPLICATION** operation, termed as **FORGET** gate
  - As Sigmoid would result in a value between 0 and 1, so <img src="https://render.githubusercontent.com/render/math?math=f_{t}"> will control what percentage of **Previous** long term state to pass to the **Current** long term state
<img src="images/forget_gate.png">
