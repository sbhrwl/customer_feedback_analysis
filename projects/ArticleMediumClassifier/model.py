# %%
import numpy as np 
import pandas as pd 
import nltk         ## for text
import re           ## regular expression
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer  ## it converts the word into its base form

# %%
## Dataset

df = pd.read_csv('tennis.csv')

# %%
df['article_text'][0]

# %%
from nltk.tokenize import sent_tokenize
sentences = []
for s in df['article_text']:
    sentences.append(sent_tokenize(s))

sentences = sentences[0]

# %%
sentences

# %%
w_d = {}

clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
clean_sentences = [s.lower() for s in clean_sentences]
stop_words = stopwords.words('english')
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

# %%
sentence_vectors = []
for i in clean_sentences:
    if len(i) != 0:
        v = sum([w_d.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
    else:
        v = np.zeros((100,))
    sentence_vectors.append(v)


# %%
sim_mat = np.zeros([len(sentences), len(sentences)])
from sklearn.metrics.pairwise import cosine_similarity
for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i != j:
            sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

# %%
import networkx as nx

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)

# %%
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
for i in range(5):
    print("ARTICLE:")
    print(df['article_text'][i])
    print('\n')
    print("SUMMARY:")
    print(ranked_sentences[i][1])
    print('\n')

# %%
