import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes
from sklearn.pipeline import Pipeline


def tfidf_with_logistic(X_train, y_train):
    print("Start training with tfidf_with_logistic_regression")
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

    tfidf_logistic_pipeline.fit(X_train, y_train)
    return tfidf_logistic_pipeline


def tfidf_with_MultinomialNB(X_train, y_train):
    print("Start training with tfidf_with_MultinomialNB")
    model_path = 'artifacts/model-artifacts'
    tf_idf = TfidfVectorizer(ngram_range=(1, 2),
                             max_features=50000,
                             min_df=2)
    # tf_idf.fit(X_train)
    #
    # # saving vector for prediction
    # with open(model_path + '/vectorizer.pickle', 'wb') as f:
    #     pickle.dump(tf_idf, f)
    #
    # X_vector = tf_idf.transform(X_train)
    #
    # model = naive_bayes.MultinomialNB()
    # model.fit(X_vector, y_train)
    # svm_svc = svm.SVC(C=1.0, kernel='gaussian', degree=3, gamma='auto')
    nb = naive_bayes.MultinomialNB()

    tfidf_multinomialNB_pipeline = Pipeline([('tf_idf', tf_idf),
                                             ('nb', nb)])
    tfidf_multinomialNB_pipeline.fit(X_train, y_train)
    with open(model_path + '/tfidf_multinomialNB.sav', 'wb') as f:
        pickle.dump(tfidf_multinomialNB_pipeline, f)

    return tfidf_multinomialNB_pipeline
