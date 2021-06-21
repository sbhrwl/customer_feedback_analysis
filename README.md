# NLP: Sentiment Analysis Project

# E2E Project
* [Overview](#overview)
* [Project Setup](#project-setup)
* [DVC Initialisation](#dvc-initialisation)
* [Pipeline](#pipeline)

# Overview
The project details about basic startegy to adopt for NLP models.
Various stages implemented in the project
* Saving dataset (save_raw_data.py)
* Feature processing
  * Clean data
    * Remove Punctuations
    * Tokenize data
    * Remove Stop words
  * Enhancing dataset
    * Stemming
    * Lemmitisation
  * Create Vectors
    * Count Vectorizer
    * Count Vectorizer with **n** grams
    * TfIdf
  * Add new features
    * Length of Comment
    * Percentage of Punctuations
* Analysis of Features
* Create Word Cloud
* Training and Evaluating Model
  * Generate Confusion Matrix
* Pycaret automl

# Project Setup
* Create environment
```bash
conda create -n nlp-project python=3.7 -y
```
* Activate environment
```bash
conda activate nlp-project
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

# DVC Initialisation
* Download the data from 
  * https://drive.google.com/drive/folders/18zqQiCJVgF7uzXgfbIJ-04zgz1ItNfF5?usp=sharing
* Initialise DVC
```bash
dvc init 
```
* Add files to dvc
```bash
dvc add data_given/dataset_reviews.csv
```
```bash
git add .
```
```bash
git commit -m "first commit"
```
* Oneliner updates  for readme
```bash
git add . && git commit -m "update Readme.md"
```
```bash
git remote add origin https://github.com/sbhrwl/nlp-project.git
git branch -M main
git push origin main
```
## Executing Pipeline
```bash
dvc repro
```

# Pipeline
## Stage 1: save_raw_data
### Results
* Cleaned data: dataset/raw/dataset_raw.csv

## Stage 2: clean_data
### Results
* Cleaned data: dataset/processed/basic_cleaning/dataset_no_punctuations.csv
* Tokenised data: dataset/processed/basic_cleaning/dataset_tokenised.csv
* Stop words removed: dataset/processed/basic_cleaning/dataset_no_stop_words.csv

## Stage 3: perform_stemming_lemmatization
### Results
* Porter Stemmer data: dataset/processed/stemmed_lemmatised/dataset_stemmed.csv
* WordNet Lemmatised data: dataset/processed/stemmed_lemmatised/dataset_lemmatised.csv

## Stage 4: text_vectorization
### Results
* Count Vectorizer data: dataset/processed/vectorized/dataset_count_vectorized.csv
* Count Vectorizer with N grams data: dataset/processed/vectorized/dataset_ngram_count_vectorized.csv
* TfIdf Vectorized data: dataset/processed/vectorized/dataset_tfidf_vectorised.csv

## Stage 5: create_features
### Results
* Create feature as Length of Comment
* Create feature as Percentage of Punctuations in a comment

## Stage 5: feature_analysis
### Results
* Length of Comment analysis: artifacts/eda-artifacts/histograms/message_length_analysis.png
* Length of Comment Logarithmic analysis: artifacts/eda-artifacts/histograms/message_length_logarithmic_analysis.png
* Percentage of Punctuations in a comment analysis: artifacts/eda-artifacts/histograms/punctuation_percent_analysis.png

## Stage 6: create_word_cloud
### Results
* Word Cloud for Comments and Rating as 4: artifacts/eda-artifacts/word_cloud/word_cloud_good.png

## Stage 7: train_and_evaluate
Create a pipeline with TfIdf and Logistic Regression, followed by evaluation of results
### Results
* Confusion Matrix: artifacts/model-artifacts/confusion_matrix_analysis.png

## Stage 8: pycaret_auto_ml
Topic Modelling with LDA using Pycaret together with MLflow
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts/mlflow-artifacts --host 0.0.0.0 -p 1234
```
### Results
Verify Exeperiment results on Ml flow UI: localhost:1234
