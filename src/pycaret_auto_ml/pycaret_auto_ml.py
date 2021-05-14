import pandas as pd
from pycaret.nlp import *
import sys
# sys.path.insert(1, './src/get_parameters')
sys.path.append('./src/get_parameters')
from get_parameters import get_parameters


def pycaret_auto_ml(dataframe):
    exp_nlp = setup(data=dataframe, target='Sentiment')


if __name__ == "__main__":
    config = get_parameters()
    dataset_raw_path = config["save_raw_data"]["dataset_raw"]
    confusion_matrix_analysis_path = config["feature_analysis"]["confusion_matrix_analysis"]
    df = pd.read_csv(dataset_raw_path, sep=",", encoding='utf-8', usecols=['Sentiment', 'Comment'])
    pycaret_auto_ml(df)
