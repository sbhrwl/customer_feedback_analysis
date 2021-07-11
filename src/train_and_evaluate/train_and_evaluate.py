import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from matplotlib import pyplot as plt
import eli5
from src.get_parameters.get_parameters import get_parameters

if __name__ == "__main__":
    config = get_parameters()
    data_path = config["feature_processing"]["dataset_with_new_features"]
    confusion_matrix_analysis_path = config["feature_analysis"]["confusion_matrix_analysis"]
    df = pd.read_csv(data_path, sep=",", encoding='utf-8',
                     usecols=['Review', 'Comment', 'Lemmatized_data', 'Message_length', 'Punctuation_Percent'])
    df = df
    print(df.head())
    tf_idf = TfidfVectorizer(ngram_range=(1, 2),
                             max_features=50000,
                             min_df=2)

    lr = LogisticRegression(C=1,
                            n_jobs=4,
                            solver='lbfgs',  # LBFGS is alternative of Gradient Descent to minimize Cost
                            random_state=17,
                            verbose=1)

    tfidf_logistic_pipeline = Pipeline([('tf_idf', tf_idf),
                                        ('lr', lr)])

    # X = df[['Lemmatized_data', 'Message_length', 'Punctuation_Percent']]
    X = df['Lemmatized_data']
    y = df['Review']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17)

    tfidf_logistic_pipeline.fit(X_train, y_train)

    y_predicted = tfidf_logistic_pipeline.predict(X_test)
    y_probability = tfidf_logistic_pipeline.predict_proba(X_test)

    # Accuracy Score
    accuracy_score(y_test, y_predicted)

    cm = confusion_matrix(y_predicted, y_test)
    print(cm)
    sns.heatmap(cm, annot=True)
    plt.savefig(confusion_matrix_analysis_path)
    print(classification_report(y_predicted, y_test))

    # show_weights works with Notebook
    # print(eli5.show_weights(estimator=tfidf_logistic_pipeline.named_steps['lr'],
    #                   vec=tfidf_logistic_pipeline.named_steps['tf_idf']))
