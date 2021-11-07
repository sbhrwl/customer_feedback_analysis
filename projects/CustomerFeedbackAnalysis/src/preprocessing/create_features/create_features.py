import pandas as pd
import numpy as np
import string
from matplotlib import pyplot
from src.get_parameters.get_parameters import get_parameters


def punctuation_count(text):
    count = sum([1 for c in text if c in string.punctuation])
    return 100*count/len(text)


def check_distribution(dataframe):
    bins = np.linspace(0, 100, 30)  # Assumption, Max Length  of message is 100 and create 30 bins
    for i in [2,3,4]:
        pyplot.hist((dataframe['Message_length'])**1/i, bins)
        # pyplot.hist((dataframe['Punctuation_Percent'])**1/i, bins)
        pyplot.title(f'Transform=1/{i}')
        pyplot.show()


if __name__ == "__main__":
    config = get_parameters()
    # data_path = config["save_raw_data"]["dataset_raw"]
    data_path = config["feature_processing"]["dataset_lemmatised"]
    dataset_with_new_features_path = config["feature_processing"]["dataset_with_new_features"]
    df = pd.read_csv(data_path, sep=",", encoding='utf-8')
    df['Message_length'] = df['Comment'].apply(lambda x: len(x))
    df['Punctuation_Percent'] = df['Comment'].apply(lambda x: punctuation_count(x))
    df.to_csv(dataset_with_new_features_path, sep=",", index=False)
    # check_distribution(df)
