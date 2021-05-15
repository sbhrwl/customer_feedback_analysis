import pandas as pd
import mlflow
from pycaret.nlp import *
import sys
# sys.path.insert(1, './src/get_parameters')
sys.path.append('./src/get_parameters')
from get_parameters import get_parameters


def topic_modelling_pycaret(dataframe):
    # https://pycaret.org/nlp/

    mlflow.set_tracking_uri('http://localhost:1234')
    exp_nlp = setup(data=dataframe, target='Comment', log_experiment=True, experiment_name='LDA-Topic Modeling')
    # print(exp_nlp)
    # Install: pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz
    lda = create_model('lda')
    # print(lda)
    lda_df = assign_model(lda)
    # print(lda_df.head())
    # plot_model(lda)
    evaluate_model(lda)
    tuned_lda = tune_model(model='lda', supervised_target='Sentiment')
    # print(tuned_lda)


if __name__ == "__main__":
    config = get_parameters()
    dataset_raw_path = config["save_raw_data"]["dataset_raw"]
    df = pd.read_csv(dataset_raw_path, sep=",", encoding='utf-8', usecols=['Comment', 'Sentiment'])
    topic_modelling_pycaret(df)
