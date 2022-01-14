# Customer feedback analysis
Utility performs survey time to time to get feedback from the customers. Build a machine learning system, to automatically classify the feedback received from the customer

* [Model Pipeline](#model-pipeline)
* [AutoML with Pycaret and MlFlow](#automl-with-pycaret-and-mlflow)
* [Project Setup](#project-setup)
* [Training and Prediction endpoint](#training-and-prediction-endpoint)

# Model Pipeline
## Stage 1: Save raw data (save_raw_data.py)
### Results
* Raw data: dataset/raw/dataset_raw.csv

## Stage 2: Basic cleaning operations (clean_data.py)
* Original Feedback
  ```
  "It offers all your essential services in one, and sells energy bundled with broadband, home phone, mobile, home insurance and boiler care."
  ```
* Punctuations removed
  ```
  "it offers all your essential services in one and sells energy bundled with broadband home phone mobile home insurance and boiler care"
  ```
* Tokenized feedback
  ```
  ['it', 'offers', 'all', 'your', 'essential', 'services', 'in', 'one', 'and', 'sells', 'energy', 'bundled', 'with', 'broadband', 'home', 'phone', 'mobile', 'home', 'insurance', 'and', 'boiler', 'care']
  ```
* Stop words removed from feedback
  ```
  ['offers', 'essential', 'services', 'one', 'sells', 'energy', 'bundled', 'broadband', 'home', 'phone', 'mobile', 'home', 'insurance', 'boiler', 'care']
  ```

### Results
* Cleaned data: dataset/processed/basic_cleaning/dataset_no_punctuations.csv
* Tokenised data: dataset/processed/basic_cleaning/dataset_tokenised.csv
* Stop words removed: dataset/processed/basic_cleaning/dataset_no_stop_words.csv

## Stage 3: Perform stemming and Lemmatization (perform_stemming_lemmatization.py)
* Stemmed feedback
  ```
  ['offer', 'essenti', 'servic', 'one', 'sell', 'energi', 'bundl', 'broadband', 'home', 'phone', 'mobil', 'home', 'insur', 'boiler', 'care']
  ```
* Lemmatized feedback
  ```
  ['offer', 'essential', 'service', 'one', 'sell', 'energy', 'bundled', 'broadband', 'home', 'phone', 'mobile', 'home', 'insurance', 'boiler', 'care']
  ```
### Results
* Porter Stemmer data: dataset/processed/stemmed_lemmatised/dataset_stemmed.csv
* WordNet Lemmatised data: dataset/processed/stemmed_lemmatised/dataset_lemmatised.csv

## Stage 4: Text vectorization (text_vectorization.py)
The feedback is now converted into vectors. These vectors can now be fed to model for training
### Results
* Count Vectorizer data: dataset/processed/vectorized/dataset_count_vectorized.csv
* Count Vectorizer with N grams data: dataset/processed/vectorized/dataset_ngram_count_vectorized.csv
* TfIdf Vectorized data: dataset/processed/vectorized/dataset_tfidf_vectorised.csv

## Stage 5: Create new features (create_features.py)
It is good practice to add meaningful features to the dataset. Such features may help model to train better with this extra information
### Results
* Create feature as Length of Comment
* Create feature as Percentage of Punctuations in a comment

## Stage 5: Feature analysis (feature_analysis.py)
### Results
* Length of Comment analysis: artifacts/eda-artifacts/histograms/message_length_analysis.png
* Length of Comment Logarithmic analysis: artifacts/eda-artifacts/histograms/message_length_logarithmic_analysis.png
* Percentage of Punctuations in a comment analysis: artifacts/eda-artifacts/histograms/punctuation_percent_analysis.png

## Stage 6: Create word cloud (create_word_cloud.py)
### Results
* Word Cloud for Comments and Rating as 4: artifacts/eda-artifacts/word_cloud/word_cloud_good.png

## Stage 7: Train model
Create a pipeline with TfIdf and Logistic Regression
```
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
```

## Stage 8: Evaluate model
### Results
* Confusion Matrix: artifacts/model-artifacts/confusion_matrix_analysis.png

# AutoML with Pycaret and MlFlow
Topic Modelling with LDA using Pycaret together with MLflow
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts/mlflow-artifacts --host 0.0.0.0 -p 1234
```
### Results
Verify Exeperiment results on Ml flow UI: localhost:1234

# Project Setup
* Create environment
```bash
conda create -n customer-feedback python=3.7 -y
```
* Activate environment
```bash
conda activate customer-feedback
```
* Install the requirements
```bash
pip install -r requirements.txt
```
* GIT initialisation
  If you plan to add changes on top of the project
  * Create your project in the Github, note down repository link
  * Perform a commit
```bash
git init
git add .
git commit -m "first commit"

git branch -M main
git remote add origin https://github.com/<githubAccount>/<githubRepo>
git push origin main
```

* DVC Initialisation
  * Download the data from 
    * https://drive.google.com/drive/folders/18zqQiCJVgF7uzXgfbIJ-04zgz1ItNfF5?usp=sharing
  * Initialise DVC
    ```bash
    dvc init 
    ```
  * Add files to dvc
    ```bash
    dvc add data_source/feedback.csv
    ```
## Executing Pipeline
```bash
dvc repro
```

# Training and Prediction endpoint
* localhost:5000/train
* localhost:5000/predict
