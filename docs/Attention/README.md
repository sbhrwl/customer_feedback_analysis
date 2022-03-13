# Attention
- [Paper](https://arxiv.org/pdf/1706.03762.pdf)
- [Neural Machine Translation Model-Mechanics of Seq2seq Models With Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

## Self Attention
 RNN, LSTM and GRU  
 
 Performance degrader for long sequences
 Remember shorter context
 
 ## Attention
 Idea is to build a "context" vector 
 Find context at any point of time of a sequence
 Diagram  from notes
 
 Implemented this block
 Building blocks of attention
 
 Reweighing  (remove Noise)
 Normalization
 Dot Product
 
 Attention Block connects 2 Blocks - Encoder , Decoder
 
 King vector  , context vector will be more meaningful
 
 
 
 Context Building 
 
 Normalize
 
 Transform
 
 Dot product
 
 Reweighing 
 
 Filters data text is far away
 Amplify data which is closer
 Smoother data
 Resulting vector can be fed to algorithm now
 
 Reweighing based on proximity/distance does not confirm to linguistics.
 Tommy can be annoying but he is a good dog.
 
 
 
