
# Word Embedding
- [Overview](#overview)
- [Text Vectorization](#text-vectorization)
  - [Text Preprocessing options before performing text vectorization](#text-preprocessing-options-for-before-performing-text-vectorization)
- [Strategies for Text vectorization](#strategies-for-text-vectorization)
  - [Machine learning](#machine-learning)
    - [One Hot Encoding](#one-hot-encoding)
    - [Bag of Words](#bag-of-words)
    - [TF-IDF](#tf-idf)
  - [Deep learning](#deep-learning)
    - [Keras Embedding layer](#keras-embedding-layer)
    - [Word2Vec](#word2vec)
    - [GloVe](#gloVe)
- [Libraries](#libraries)
  - [nltk](https://www.nltk.org/)
  - [TextBlob](https://colab.research.google.com/drive/11PEnYPnmi0eS9wVOXn1lSfiTxFUvUYNi?usp=sharing)
  - [SpaCy](https://colab.research.google.com/drive/1IRfOFBQ5N6_m0cMgmkzagtfCACrf8sxJ?usp=sharing)
    - [missing images](https://drive.google.com/drive/folders/1HL7EiZlpOloAlRI8Mm2jr-x9r2jqM11H)
  - [Prodi.gy](https://prodi.gy/)
- [Stanford Learn Natural Language](https://web.stanford.edu/~jurafsky/slp3/)
  - [Book](https://web.stanford.edu/~jurafsky/slp3/ed3book_jan122022.pdf)

## Overview
- Word embedding is a form of word representation that connects the human understanding of language to that of the machine. 
- Word embeddings are the distributed representations of text in an **ample dimensional space**. 
- Word embeddings are a class of techniques where the **individual word is represented as a real-valued vector in a vector space**. 
- The main idea is to use a densely distributed representation for all the words. 
- Each word is represented by a real-value vector. 
- Each word is mapped to a single vector, and **the vector values are learned in a way that resembles a neural network**

## Text Vectorization
- The neural network cannot train the original text data. 
- We need to process the text data into numerical tensors first. 
- This process is also called text vectorization.

### Text Preprocessing options before performing text vectorization
- Split text into **words**, each word is converted into a vector
- Split text into **characters**, each character is converted into a vector
- Extract **n-gram of words or characters** each n-gram is converted into a vector

## Strategies for Text vectorization
## Machine learning
### One Hot Encoding
### Bag of Words
- The Bag-of-words model (BoW model) ignores the grammar and word order of a text, and uses a set of unordered words to express a text or a document.
- Example
  - Sentence 1: `John likes to watch movies. Mary likes too.`
  - Sentence 2: `John also likes to watch football games.`
  - **Build a dictionary**
    `{"John": 1, "likes": 2, "to": 3, "watch": 4, "movies": 5, "also": 6, "football": 7, "games": 8, "Mary": 9, "too": 10}`
  - Represent the abpove 2 sentences using BOW
    - Sentence 1: `[1, 2, 1, 1, 1, 0, 0, 0, 1, 1]`
    - Sentence 2: `[1, 1, 1, 1, 0, 1, 1, 1, 0, 0]`

### Count vectorizer
- Count vectorizer is used to implement bag of words
### TF-IDF
## Deep learning
### Keras Embedding layer
- Embedding layer
```python
from keras.layers import Embedding
embedding_layer = Embedding(1000,64)
```
- Example 1
```python
from keras.models import Sequential
from keras.layers import Flatten,Dense,Embedding

model = Sequential()
#Instantiate an Embedding layer
model.add(Embedding(10000,8,input_length=20))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.summary()
```
- Example 2
```python
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten,Dense,Embedding

max_features = 10000
maxlen = 20

(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=10000)
#(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

x_train = preprocessing.sequence.pad_sequences(x_train,maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test,maxlen=maxlen)

model = Sequential()

model.add(Embedding(10000,8,input_length=maxlen))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history = model.fit(x_train,y_train,epochs=10,batch_size=32,validation_split=0.2,verbose=1)
```
- Measure Results
```python
import matplotlib.pyplot as plt

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='acc')
plt.plot(epochs,val_acc,'b',label='val_acc')
plt.legend()

plt.figure()
plt.plot(epochs,loss,'bo',label='loss')
plt.plot(epochs,val_loss,'b',label='val_loss')
plt.legend()

plt.show()
```
### Word2Vec
- [blog](https://israelg99.github.io/2017-03-23-Word2Vec-Explained/)
- [blog](https://jalammar.github.io/illustrated-word2vec/)
- [blog](https://builtin.com/machine-learning/nlp-word2vec-python)
- [blog](https://radimrehurek.com/gensim/models/word2vec.html)
### GloVe
- GloVe is a type of Word embedding. 
- The format of the GloVe word vector and word2vec is a little different from the Stanford open source code training. 
- The first line of the model trained by word2vec is: thesaurus size and dimensions, while gloVe does not
## Links
- [Flashtext better than regular expressions]https://arxiv.org/pdf/1711.00046.pdf
- [Parts of speech](https://sites.google.com/site/partofspeechhelp/)
- [Xceptor: No-code data automation platform](https://www.xceptor.com/)
