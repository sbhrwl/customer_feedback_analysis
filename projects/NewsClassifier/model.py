import pandas as pd
import re # Regular Expression
import string
from tensorflow import keras
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter


df = pd.read_csv(r'projects/NLP/NewsClassifier/data/news.csv')
# df.head()
# df.shape
print((df.target == 1).sum()) # disaster
print((df.target == 0).sum()) # no disaster


def remove_url(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)


def remove_punc(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def remove_stopwords(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)


# Count unique words
def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count


string.punctuation
pattern = re.compile(r"https?://(\S+|www)\.\S+")
for t in df.text:
    matches = pattern.findall(t)
    for match in matches:
        print("1")
        print(t)
        print("2")
        print(match)
        print("3")
        print(pattern.sub(r"", t))
    if len(matches) > 0:
        break
df['text'] = df.text.map(remove_url)        # map(lambda x: remove_url(x))
df['text'] = df.text.map(remove_punc)
df.head()

# remove stopwords
stop = set(stopwords.words("english"))
df["text"] = df.text.map(remove_stopwords)
df.text


counter = counter_word(df.text)
len(counter)
counter.most_common(5)
num_unique_words = len(counter)
train_size = int(df.shape[0] * 0.8)
train_df = df[:train_size]
val_df = df[train_size:]

train_sentence = train_df.text.to_numpy()
train_labels = train_df.target.to_numpy()
val_sentence = val_df.text.to_numpy()
val_labels = val_df.target.to_numpy()
train_sentence.shape, val_sentence.shape
train_sentence
# tokenize

from tensorflow.keras.preprocessing.text import Tokenizer

# vectorize a text corpus by turning each text into a sequence of interger
tokenizer = Tokenizer(num_words=num_unique_words)
tokenizer.fit_on_texts(train_sentence)  # fit only to training
# each word has unique index
word_index = tokenizer.word_index
# word_index
train_sequence = tokenizer.texts_to_sequences(train_sentence)
val_sequence = tokenizer.texts_to_sequences(val_sentence)
print(train_sentence[0:5])
print(train_sequence[0:5])
# pad the sentence to have the same length
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Maximum number of words in a suquence
max_length = 20

train_padded = pad_sequences(train_sequence, maxlen=max_length, padding="post", truncating="post")
val_padded = pad_sequences(val_sequence, maxlen=max_length, padding="post", truncating="post")
train_padded.shape, val_padded.shape
train_padded[0]
print(train_sentence[10])
print(train_sequence[10])
print(train_padded[10])
# Check reversing the indices

# flip key, values
reverse_word_index = dict([(idx, word) for word, idx in word_index.items()])
# reverse_word_index
def decode(sequence):
    return ' '.join([reverse_word_index.get(idx, "?") for idx in sequence])
decode_text = decode(train_sequence[10])
print(train_sequence[10])
print(decode_text)
# Create LSTM Model
from tensorflow.keras import layers

# Embedding: https://www.tensorflow.org/tutorials/text/word_embeddings
# Turns positive integers (indexes) into dense vectors of fixed size. (other approach could be one-hot-encoding)

model = keras.models.Sequential()
model.add(layers.Embedding(num_unique_words, 32, input_length=max_length))

# The layer will take as input an integer matrix of size (batch, input_length)
# and  the largest integer (i.e. word index) in the input should be no longer than num_words (vocabulary size)
# Now model.output_shape is (None, input_length, 32), where None is tha batch dimension

model.add(layers.LSTM(64, dropout=0.1))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
loss = keras.losses.BinaryCrossentropy(from_logits=False)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ['accuracy']

model.compile(loss=loss, optimizer=optim, metrics=metrics)
model.fit(train_padded, train_labels, epochs=20, validation_data=(val_padded, val_labels), verbose=2)
predictions = model.predict(train_padded)
predictions = [1 if p> 0.5 else 0 for p in predictions]
print(train_sentence[10:20])
print(train_labels[10:20])
print(predictions[10:20])
