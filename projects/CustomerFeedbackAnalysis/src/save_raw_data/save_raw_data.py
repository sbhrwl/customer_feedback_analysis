import pandas as pd
from src.get_parameters.get_parameters import get_parameters


def save_raw_data():
    config = get_parameters()
    data_path = config["data_source"]["store_location"]
    raw_data_path = config["save_raw_data"]["dataset_raw"]

    df = pd.read_csv(data_path, sep=",", encoding='utf-8')
    # Save Raw dataset to directory "raw"
    df.to_csv(raw_data_path, sep=",", index=False)


if __name__ == "__main__":
    save_raw_data()
