"""
	Newsgroup 20 sample
"""

import ncn

clf = ncn.NCN(nr_estimators=50, compressor="snappy", max_anchor_size=50000, sub_sample=.6, verbose=2)

print clf

from sklearn.datasets import fetch_20newsgroups
import numpy as np

categories = ['soc.religion.christian', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

y = twenty_train.target
X = twenty_train.data

clf.fit(X[:200]) # we fit on the first 200 documents

X = clf.transform(X[200:]) # we transform the remaining documents

print np.array(X).shape

from sklearn import cross_validation, svm
clf_model = svm.SVC(kernel="linear",degree=1,C=100.)

print clf_model
scores = cross_validation.cross_val_score(clf_model, X, y[200:], cv=3, verbose=3)

print scores.mean()