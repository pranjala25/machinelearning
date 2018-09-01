# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 14:18:29 2018

@author: ppranjal
Naive Bayes to classify 20 Newsgroup dataset
"""

#Text Classification
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
training_data = fetch_20newsgroups(subset='train',shuffle=True)
print(training_data.target_names)
print('\n'.join(training_data.data[0].split('\n')[:3]))

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
X_train_count = count_vector.fit_transform(training_data.data)
print(X_train_count.shape)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)
print(X_train_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf,training_data.target)

test_data = fetch_20newsgroups(subset='test',shuffle=True)
# User transform instead of fit_trasform, else run into Dimension mismatch error( as the vectorizer & Tfidf are already fit)
X_test_count = count_vector.transform(test_data.data)
X_test_tfidf = tfidf_transformer.transform(X_test_count)

predicted = clf.predict(X_test_tfidf)
print(np.mean(predicted == test_data.target))

#Writing all the above in less code using Pipeline
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('count',CountVectorizer(stop_words='english',ngram_range=(1,1))),('tfidf',TfidfTransformer()),('ndclf',MultinomialNB())])
text_clf.fit(training_data.data,training_data.target)
predicted = text_clf.predict(test_data.data)
print(np.mean(predicted == test_data.target))


