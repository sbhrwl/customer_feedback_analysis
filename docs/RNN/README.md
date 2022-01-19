# RNN
- [Drawbacks of CNN and ANN](#drawbacks-of-cnn-and-ann)
- [RNN Depiction](#rnn-depiction)
  - [Passing previous state to a Neuron](#passing-previous-state-to-a-neuron)
  - [Passing previous state to an ANN](#passing-previous-state-to-an-ann)
  - [Feed Forward](#feed-forward)
- [Types of RNN Configuration](#types-of-rnn-configuration)
  - [Seq to Seq](#seq-to-seq)
  - [Seq to Vector](#seq-to-vector)
  - [Vector to Seq](#vector-to-seq)
  - [Encoder-Decoder](#encoder-decoder)
- [RNN Equation and Weight Matrix](#rnn-equation-and-weight-matrix)
- [How does RNN works](#how-does-rnn-works)

## Drawbacks of CNN and ANN
- There is no memory element
- The present data is not dependent on previous data
- CNN and ANN are not good for Sequential data
  - Time series (Stock price), Text data (Language Translation), Speech (Speech Recognition)
  
## RNN Depiction
- Below picture depicts single Neuron or RNN expanded over time
- **Unrolling across time**- Bring information from `previous state` or previous timestamp (previous word or ngram)
### Passing previous state to a Neuron
<img src="images/Previous-State-Input-Neuron.png" width=200>

### Passing previous state to an ANN
<img src="images/Previous-State-Input-ANN.png" width=200>

<img src="images/RNN-Depiction.png">

## Feed Forward
<img src="images/FFNN-Depiction.png">




## Back Propogation Through Time
<img src="images/BPTT.png">

## Types of RNN Configuration
### Seq to Seq
<img src="images/Seq-to-Seq.png">

### Seq to Vector
<img src="images/Seq-to-Vector.png">

### Vector to Sequence
<img src="images/Vector-to-Sequence.png">

### Encoder-Decoder
<img src="images/Encode-Decoder.png">

## RNN Equation and Weight Matrix
### Partial derivative
<img src="images/partial-derivative.png">

### Weight Matrix
<img src="images/RNN-Equation-WeightMatrix.png">

## How does RNN works
<img src="images/How-To-Solve.png">
