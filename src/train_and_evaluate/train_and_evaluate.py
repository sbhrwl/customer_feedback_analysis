import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from matplotlib import pyplot as plt
import eli5
import sys
# sys.path.insert(1, './src/get_parameters')
sys.path.append('./src/get_parameters')
from get_parameters import get_parameters


if __name__ == "__main__":
    config = get_parameters()
    dataset_raw_path = config["save_raw_data"]["dataset_raw"]
    confusion_matrix_analysis_path = config["feature_analysis"]["confusion_matrix_analysis"]
    df = pd.read_csv(dataset_raw_path, sep=",", encoding='utf-8', usecols=['Sentiment', 'Comment'])
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

    tfidf_logit_pipeline = Pipeline([('tf_idf', tf_idf),
                                     ('lr', lr)])

    X_train, X_test, y_train, y_test = train_test_split(df['Comment'], df['Sentiment'], random_state=17)

    tfidf_logit_pipeline.fit(X_train, y_train)

    y_pred = tfidf_logit_pipeline.predict(X_test)
    y_proba = tfidf_logit_pipeline.predict_proba(X_test)

    # Accuracy Score
    accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_pred, y_test)
    print(cm)
    sns.heatmap(cm, annot=True)
    plt.savefig(confusion_matrix_analysis_path)
    print(classification_report(y_pred, y_test))

    # show_weights works with Notebook
    # print(eli5.show_weights(estimator=tfidf_logit_pipeline.named_steps['lr'],
    #                   vec=tfidf_logit_pipeline.named_steps['tf_idf']))
