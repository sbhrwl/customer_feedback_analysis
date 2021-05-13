import pandas as pd
import nltk
from nltk.stem import PorterStemmer
import string
import re
from nltk.corpus import stopwords
import sys
# sys.path.insert(1, './src/get_parameters')
sys.path.append('./src/get_parameters')
from get_parameters import get_parameters
nltk.download('wordnet')


def clean_text(text):
    text_no_punctuation = "".join([item for item in text if item not in string.punctuation])
    tokenized_text = re.split('\W+', text_no_punctuation)
    stop_words = stopwords.words('english')
    text_clean = [word for word in tokenized_text if word not in stop_words]
    return text_clean


def perform_stemming(text):
    ps = PorterStemmer()
    # return a list [] with stemmed data ps.stem(word)
    stemmed_text = [ps.stem(word) for word in text]
    return stemmed_text


def perform_lemmatisation(text):
    wn = nltk.WordNetLemmatizer()
    lemmatised_text = [wn.lemmatize(word) for word in text]
    return lemmatised_text


if __name__ == "__main__":
    config = get_parameters()
    data_path = config["save_raw_data"]["dataset_raw"]
    df = pd.read_csv(data_path, sep=",", encoding='utf-8')

    df['Stop_Words_Removed'] = df['Comment'].apply(lambda x: clean_text(x.lower()))

    df['Stemmed_data'] = df['Stop_Words_Removed'].apply(lambda x: perform_stemming(x))
    dataset_stemmed_data_path = config["feature_processing"]["dataset_stemmed"]
    df.to_csv(dataset_stemmed_data_path, sep=",", index=False)

    df['Lemmatized_data'] = df['Stop_Words_Removed'].apply(lambda x: perform_lemmatisation(x))
    dataset_lemmatised_data_path = config["feature_processing"]["dataset_lemmatised"]
    df.to_csv(dataset_lemmatised_data_path, sep=",", index=False)
