import pandas as pd
import seaborn as sns
import eli5
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from src.get_parameters.get_parameters import get_parameters
from src.train_and_evaluate.training_options import tfidf_with_logistic, tfidf_with_MultinomialNB


def train_and_evaluate():
    config = get_parameters()
    data_path = config["feature_processing"]["dataset_with_new_features"]
    confusion_matrix_analysis_path = config["feature_analysis"]["confusion_matrix_analysis"]

    df = pd.read_csv(data_path, sep=",", encoding='utf-8',
                     usecols=['Review', 'Comment', 'Lemmatized_data', 'Message_length', 'Punctuation_Percent'])
    print(df.head())

    # X = df[['Lemmatized_data', 'Message_length', 'Punctuation_Percent']]
    X = df['Lemmatized_data']
    y = df['Review']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17)

    # model = tfidf_with_logistic(X_train, y_train)
    model = tfidf_with_MultinomialNB(X_train, y_train)

    y_predicted = model.predict(X_test)
    # y_probability = model.predict_proba(X_test)

    # Accuracy Score
    accuracy_score(y_test, y_predicted)

    cm = confusion_matrix(y_predicted, y_test)
    print(cm)
    sns.heatmap(cm, annot=True)
    plt.savefig(confusion_matrix_analysis_path)
    print(classification_report(y_predicted, y_test))

    # show_weights: works with Notebook
    # print(eli5.show_weights(estimator=model.named_steps['lr'],
    #                         vec=model.named_steps['tf_idf']))


if __name__ == "__main__":
    train_and_evaluate()
