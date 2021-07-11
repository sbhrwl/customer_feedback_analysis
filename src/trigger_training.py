from src.save_raw_data.save_raw_data import save_raw_data
from src.preprocessing.clean_data.clean_data import clean_data
from src.preprocessing.perform_stemming_lemmatization.perform_stemming_lemmatization import perform_stemming_lemmatization
from src.train_and_evaluate.train_and_evaluate import train_and_evaluate


def trigger_training():
    save_raw_data()
    clean_data()
    perform_stemming_lemmatization()
    train_and_evaluate()


if __name__ == "__main__":
    trigger_training()
