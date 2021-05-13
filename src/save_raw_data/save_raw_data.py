import pandas as pd
import sys
# sys.path.insert(1, './src/get_parameters')
sys.path.append('./src/get_parameters')
from get_parameters import get_parameters


def save_raw_data():
    config = get_parameters()
    data_path = config["data_source"]["store_location"]
    df = pd.read_csv(data_path, sep=",", encoding='utf-8')


    # Save Raw dataset to directory "raw"
    raw_data_path = config["save_raw_data"]["dataset_raw"]
    df.to_csv(raw_data_path, sep=",", index=False)


if __name__ == "__main__":
    save_raw_data()
