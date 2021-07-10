import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from src.get_parameters.get_parameters import get_parameters
nltk.download('stopwords')


def remove_punctuation(text):
    # Returns list
    # text_no_punctuation = [i for i in text if i not in string.punctuation]

    # Use join to return words
    text_no_punctuation = "".join([i for i in text if i not in string.punctuation])
    return text_no_punctuation


def tokenize_data(text):
    # Split on non word (capital W) characters
    tokens = re.split('\W+', text)
    return tokens


def remove_stop_words(tokenized_text):
    text_clean = [word for word in tokenized_text if word not in stop_words]
    return text_clean


if __name__ == "__main__":
    config = get_parameters()
    data_path = config["save_raw_data"]["dataset_raw"]
    df = pd.read_csv(data_path, sep=",", encoding='utf-8')

    df['NO_Punctuations'] = df['Comment'].apply(lambda x: remove_punctuation(x.lower()))
    dataset_no_punctuations_data_path = config["feature_processing"]["dataset_no_punctuations"]
    df.to_csv(dataset_no_punctuations_data_path, sep=",", index=False)

    df['Tokenized_Data'] = df['NO_Punctuations'].apply(lambda x: tokenize_data(x))
    dataset_tokenised_data_path = config["feature_processing"]["dataset_tokenised"]
    df.to_csv(dataset_tokenised_data_path, sep=",", index=False)

    stop_words = stopwords.words('english')
    df['Stop_Words_Removed'] = df['Tokenized_Data'].apply(lambda x: remove_stop_words(x))
    dataset_no_stop_words_data_path = config["feature_processing"]["dataset_no_stop_words"]
    df.to_csv(dataset_no_stop_words_data_path, sep=",", index=False)
