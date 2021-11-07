import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv('projects/data/', sep='\t', names=['label', 'message'])
df.head()
df.shape

ps = PorterStemmer()
corpus = []
for i in range(len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['message'][i])
    review = review.lower()
    review = nltk.word_tokenize(review)
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()
X.shape
y = pd.get_dummies(df['label'])
y = y.iloc[:, 1]
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred = spam_detect_model.predict(X_test)
a = {'Actual': y_test, 'Predicted': y_pred}
pd.DataFrame(a).tail(50)

confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
