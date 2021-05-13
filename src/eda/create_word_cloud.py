import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from matplotlib import pyplot as plt
import sys
# sys.path.insert(1, './src/get_parameters')
sys.path.append('./src/get_parameters')
from get_parameters import get_parameters


def create_word_cloud(dataframe, file_to_create):
    wordcloud = WordCloud(background_color='black',
                          stopwords=STOPWORDS,
                          max_words=200,
                          max_font_size=100,
                          random_state=17,
                          width=800,
                          height=400)

    plt.figure(figsize=(16, 12))
    wordcloud.generate(str(dataframe.loc[dataframe['Sentiment'] == 4, 'Comment']))
    plt.imshow(wordcloud)
    plt.title('Word Cloud for Good Reviews')
    plt.savefig(file_to_create)


if __name__ == "__main__":
    config = get_parameters()
    dataset_with_new_features_path = config["feature_processing"]["dataset_with_new_features"]
    word_cloud_bad_file_path = config["feature_analysis"]["word_cloud_bad"]
    word_cloud_neutral_file_path = config["feature_analysis"]["word_cloud_neutral"]
    word_cloud_good_file_path = config["feature_analysis"]["word_cloud_good"]
    df = pd.read_csv(dataset_with_new_features_path, sep=",", encoding='utf-8')
    create_word_cloud(df, word_cloud_good_file_path)
