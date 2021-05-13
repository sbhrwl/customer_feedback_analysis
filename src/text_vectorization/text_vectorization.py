import pandas as pd
import string
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import sys
# sys.path.insert(1, './src/get_parameters')
sys.path.append('./src/get_parameters')
from get_parameters import get_parameters
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text):
    text_no_punctuation = "".join([item for item in text if item not in string.punctuation])
    tokenized_text = re.split('\W+', text_no_punctuation)
    stop_words = stopwords.words('english')
    text_clean = [word for word in tokenized_text if word not in stop_words]
    return text_clean


def clean_text_for_Ngrams(text):
    txt_no_punctuation = "".join([item for item in text if item not in string.punctuation])
    tokenized_text = re.split('\W+', txt_no_punctuation)
    text_clean = " ".join([ps.stem(word) for word in tokenized_text if word not in stop_words])
    return text_clean


def text_vectorization(dataframe, file_to_create):
    cv = CountVectorizer(analyzer=clean_text)
    X = cv.fit_transform(dataframe['Comment'])
    X.shape
    df_count_vectorised = pd.DataFrame(X.toarray(), columns=cv.get_feature_names())
    df_count_vectorised.to_csv(file_to_create, sep=",", index=False)


def text_ngram_vectorization(dataframe, file_to_create):
    cv = CountVectorizer(ngram_range=(2, 2))
    X = cv.fit_transform(dataframe['Comment'])
    X.shape
    df_count_vectorised = pd.DataFrame(X.toarray(), columns=cv.get_feature_names())
    df_count_vectorised.to_csv(file_to_create, sep=",", index=False)


def text_tfidf_vectorization(dataframe, file_to_create):
    tfidf_vectorizer = TfidfVectorizer(clean_text_for_Ngrams)
    X = tfidf_vectorizer.fit_transform(dataframe['Comment'])
    X.shape
    df_count_vectorised = pd.DataFrame(X.toarray(), columns=tfidf_vectorizer.get_feature_names())
    df_count_vectorised.to_csv(file_to_create, sep=",", index=False)


if __name__ == "__main__":
    config = get_parameters()
    data_path = config["save_raw_data"]["dataset_raw"]
    dataset_count_vectorized_data_path = config["feature_processing"]["dataset_count_vectorized"]
    dataset_ngram_count_vectorized = config["feature_processing"]["dataset_ngram_count_vectorized"]
    dataset_tfidf_vectorised = config["feature_processing"]["dataset_tfidf_vectorised"]
    df = pd.read_csv(data_path, sep=",", encoding='utf-8')

    df['Stop_Words_Removed'] = df['Comment'].apply(lambda x: clean_text(x.lower()))
    text_vectorization(df, dataset_count_vectorized_data_path)
    text_ngram_vectorization(df, dataset_ngram_count_vectorized)
    text_tfidf_vectorization(df, dataset_tfidf_vectorised)
