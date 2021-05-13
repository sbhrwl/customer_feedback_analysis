import pandas as pd
import string
import sys
# sys.path.insert(1, './src/get_parameters')
sys.path.append('./src/get_parameters')
from get_parameters import get_parameters


def punctuation_count(text):
    count = sum([1 for c in text if c in string.punctuation])
    return 100*count/len(text)


if __name__ == "__main__":
    config = get_parameters()
    data_path = config["save_raw_data"]["dataset_raw"]
    dataset_with_new_features_path = config["feature_processing"]["dataset_with_new_features"]
    df = pd.read_csv(data_path, sep=",", encoding='utf-8')
    df['Message_length'] = df['Comment'].apply(lambda x: len(x))
    df['Punctuation_Percent'] = df['Comment'].apply(lambda x: punctuation_count(x))
    df.to_csv(dataset_with_new_features_path, sep=",", index=False)
