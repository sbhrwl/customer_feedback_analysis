# NLP: Sentiment Analysis Project

## Setup 

Create env 
```bash
conda create -n nlp-project python=3.7 -y
```

Activate env
```bash
conda activate nlp-project
```

Install the req
```bash
pip install -r requirements.txt
```

```bash
pip uninstall lazypredict
```

## Commit 
If you plan to add changes on top of the project

a. Create your project in the Github, note down repository link

b. Perform a commit

```bash
git add .
git commit -m "first commit"

git branch -M main
git remote add origin https://github.com/<githubAccount>/<githubRepo>
git push origin main
```

## DVC Initialisation
i. Copy dataset to "dataset" directory

ii. Initialise DVC (dvc init)

iii. Add files to dvc: dvc add -R <directoryName>

iv. Store files on Google drive: dvc remote add -d storage gdrive://1jdxYH7rP7fgMXWPx_S062rbHAwee66

    ONLY ID part of the https://drive.google.com/drive/u/0/folders/1jdxYH7rP7fgMXWPx_S062rbHAwee66
    
v. Push files to Google drive: dvc push

vi. While authenticating, make sure you copy entire URL 

```bash
dvc init
dvc add -R <directoryName>
dvc remote add -d storage gdrive://1jdxYH7rP7fgMXWPx_S062rbHAwee66
dvc push
```
After dvc push, check the files in your Google drive

## Executing Pipeline

```bash
dvc repro
```

# Pipeline

## Stage 1: save_raw_data

### Results

1. Cleaned data: dataset/raw/dataset_raw.csv

## Stage 2: clean_data

### Results

1. Cleaned data: dataset/processed/basic_cleaning/dataset_no_punctuations.csv

2. Tokenised data: dataset/processed/basic_cleaning/dataset_tokenised.csv

3. Stop words removed: dataset/processed/basic_cleaning/dataset_no_stop_words.csv

## Stage 3: perform_stemming_lemmatization

### Results

1. Porter Stammer data: dataset/processed/stemmed_lemmatised/dataset_stemmed.csv

2. WordNet Lemmatised data: dataset/processed/stemmed_lemmatised/dataset_lemmatised.csv

## Stage 4: text_vectorization

### Results

1. Count Vectorizer data: dataset/processed/vectorized/dataset_count_vectorized.csv

2. Count Vectorizer with N grams data: dataset/processed/vectorized/dataset_ngram_count_vectorized.csv

3. TfIdf Vectorized data: dataset/processed/vectorized/dataset_tfidf_vectorised.csv

## Stage 5: create_features

### Results

1. Create feature as Length of Comment

2. Create feature as Percentage of Punctuations in a comment

## Stage 5: feature_analysis

### Results

 1. Length of Comment analysis: artifacts/eda-artifacts/histograms/message_length_analysis.png
 
 2. Length of Comment Logarithmic analysis: artifacts/eda-artifacts/histograms/message_length_logarithmic_analysis.png
 
 3. Percentage of Punctuations in a comment analysis: artifacts/eda-artifacts/histograms/punctuation_percent_analysis.png


## Stage 6: create_word_cloud

### Results

 1. Word Cloud for Comments and Rating as 4: artifacts/eda-artifacts/word_cloud/word_cloud_good.png

## Stage 7: train_and_evaluate

Create a pipeline with TfIdf and Logistic Regression, followed by evaluation of results

### Results

 1. Confusion Matrix: artifacts/model-artifacts/confusion_matrix_analysis.png
