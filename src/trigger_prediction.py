import pandas as pd
import pickle
from src.get_parameters.get_parameters import get_parameters
from src.save_raw_data.save_raw_data import save_raw_data
from src.preprocessing.clean_data.clean_data import clean_data
from src.preprocessing.perform_stemming_lemmatization.perform_stemming_lemmatization import perform_stemming_lemmatization


def load_model(model):
    with open(model, 'rb') as f:
        return pickle.load(f)


def trigger_prediction():
    config = get_parameters()
    dataset_lemmatised = config["feature_processing"]["dataset_lemmatised"]
    model_path = 'artifacts/model-artifacts'

    save_raw_data()
    clean_data()
    perform_stemming_lemmatization()

    df = pd.read_csv(dataset_lemmatised, sep=",", encoding='utf-8')

    prediction_model = load_model(model_path + '/tfidf_multinomialNB.sav')
    result = list(prediction_model.predict(df['Lemmatized_data']))
    df['Predicted_Output'] = result
    df['Actual_Output'] = df['Review']
    df.to_csv("Predictions.csv", header=True, mode='a+')


if __name__ == "__main__":
    trigger_prediction()
