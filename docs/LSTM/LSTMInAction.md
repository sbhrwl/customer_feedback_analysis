# LSTM in action

## Consider different states coming in and going out from a LSTM cell
<img src="images/states_lstm.png">

## Long Term memory
### Forget gate
- **Previous Short** term state and **Current Input feature** will go through an **ANN**
- Result of ANN will go through **Sigmoid** function <img src="https://render.githubusercontent.com/render/math?math=f_{t}">
- <img src="https://render.githubusercontent.com/render/math?math=f_{t}"> will now go through **MULTIPLICATION** operation, termed as **FORGET** gate
  - As Sigmoid would result in a value between `0 and 1`, so <img src="https://render.githubusercontent.com/render/math?math=f_{t}"> will control what percentage of **Previous** long term state to pass to the **Current** long term state
<img src="images/forget_gate.png">

### Input Gate
- **Previous Short** term state and **Current Input feature** will go through an **ANN**
- Result of ANN will go through **tanh** function <img src="https://render.githubusercontent.com/render/math?math=g_{t}">
  - Output of **tanh** lies between `-1 and +1`
- At the same time, we will also have <img src="https://render.githubusercontent.com/render/math?math=i_{t}">
  - <img src="https://render.githubusercontent.com/render/math?math=i_{t}"> is based on **Sigmoid** function
- Now, at the **Input Gate**, <img src="https://render.githubusercontent.com/render/math?math=g_{t}"> can be **+ve or -ve**
  - <img src="https://render.githubusercontent.com/render/math?math=g_{t}"> will be **Multiplied** with <img src="https://render.githubusercontent.com/render/math?math=i_{t}">
  - Output at Input gate would be either a **+ve** or a **-ve** number
- This number will further **add** or **subtract** information from **Long** term memory
- Finally we have built our **Long term memory**
<img src="images/lstm.png" width=500>

## Short term memory
### Output gate
- **Previous Short** term state and **Current Input feature** will go through an **ANN**
- Result of ANN will go through **Sigmoid** function <img src="https://render.githubusercontent.com/render/math?math=o_{t}">
- At the same time, **Long** term memory is passed through **tanh**
  - As Output of **tanh** lies between `-1 and +1`, this would give a **+ve** or a **-ve** number
- <img src="https://render.githubusercontent.com/render/math?math=o_{t}"> and above generated **+ve** or  **-ve** number, will now go through **MULTIPLICATION** operation
- This will produce
  - **Short term memory** <img src="https://render.githubusercontent.com/render/math?math=h_{t}"> and 
  - Output <img src="https://render.githubusercontent.com/render/math?math=y_{t}">
