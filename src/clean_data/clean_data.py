import pandas as pd
import sys
# sys.path.insert(1, './src/get_parameters')
sys.path.append('./src/get_parameters')
from get_parameters import get_parameters
import string


def remove_punctuation(text):
    # Returns list
    # text_no_punctuation = [i for i in text if i not in string.punctuation]

    # Use join to return words
    text_no_punctuation = "".join([i for i in text if i not in string.punctuation])
    return text_no_punctuation


def clean_data():
    config = get_parameters()
    data_path = config["save_raw_data"]["dataset_raw"]
    df = pd.read_csv(data_path, sep=",", encoding='utf-8')

    df['No_Punctuations'] = df['Comment'].apply(lambda x: remove_punctuation(x))
    # Save Raw dataset to directory "raw"
    dataset_no_punctuations_data_path = config["feature_processing"]["dataset_no_punctuations"]
    df.to_csv(dataset_no_punctuations_data_path, sep=",", index=False)


if __name__ == "__main__":
    clean_data()
