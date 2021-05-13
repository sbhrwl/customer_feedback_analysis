import pandas as pd
from matplotlib import pyplot
import numpy as np
# %matplotlib inline
import sys
# sys.path.insert(1, './src/get_parameters')
sys.path.append('./src/get_parameters')
from get_parameters import get_parameters


def message_length_analysis(dataframe, file_to_create):
    bins = np.linspace(0, 100, 30)  # Assumption, Max Length  of message is 100 and create 30 bins
    pyplot.hist(dataframe[dataframe['Sentiment'] == 0]['Message_length'], bins, label='bad')
    pyplot.hist(dataframe[dataframe['Sentiment'] == 2]['Message_length'], bins, label='neutral')
    pyplot.hist(dataframe[dataframe['Sentiment'] == 4]['Message_length'], bins, label='good')
    pyplot.legend(loc='upper right')
    pyplot.savefig(file_to_create)


def punctuation_percent_analysis(dataframe, file_to_create):
    bins = np.linspace(0, 100, 30)
    pyplot.hist(dataframe[dataframe['Sentiment'] == 0]['Punctuation_Percent'], bins, label='bad')
    pyplot.hist(dataframe[dataframe['Sentiment'] == 2]['Punctuation_Percent'], bins, label='neutral')
    pyplot.hist(dataframe[dataframe['Sentiment'] == 4]['Punctuation_Percent'], bins, label='good')
    pyplot.legend(loc='upper right')
    pyplot.savefig(file_to_create)


if __name__ == "__main__":
    config = get_parameters()
    dataset_with_new_features_path = config["feature_processing"]["dataset_with_new_features"]
    message_length_analysis_file_path = config["feature_analysis"]["message_length_analysis"]
    punctuation_percent_analysis_file_path = config["feature_analysis"]["punctuation_percent_analysis"]
    df = pd.read_csv(dataset_with_new_features_path, sep=",", encoding='utf-8')
    message_length_analysis(df, message_length_analysis_file_path)
    punctuation_percent_analysis(df, punctuation_percent_analysis_file_path)
