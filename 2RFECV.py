import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import warnings
import tensorflow
import tensorflow.keras
from tensorflow.keras.applications import imagenet_utils
from sklearn.svm import SVC
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR


fenlei = 'cough nocough'.split()
file_name = 'data/pigCoughThermalv2/acoustic_data.csv'

data = pd.read_csv(file_name)
data = data.drop(['filename'],axis=1)
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

scaler = StandardScaler()


X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


svm_model_baseDir = './weight/pigCoughThermalv2SVM/'
pathlib.Path(svm_model_baseDir).mkdir(parents=True, exist_ok=True)

feat_labels = data.columns
estimator = SVC(kernel="linear")

selector = RFECV(estimator, step=1, cv=5, scoring='f1')
selector = selector.fit(X_train, y_train)
joblib.dump(selector, svm_model_baseDir+'lab1_fold5_f1.pkl')

selector = RFECV(estimator, step=1, cv=5, scoring='accuracy')
selector = selector.fit(X_train, y_train)
joblib.dump(selector, svm_model_baseDir+'lab1_fold5_acc.pkl')

selector = RFECV(estimator, step=1, cv=5, scoring='precision')
selector = selector.fit(X_train, y_train)
joblib.dump(selector, svm_model_baseDir+'lab1_fold5_prec.pkl')

selector = RFECV(estimator, step=1, cv=5, scoring='recall')
selector = selector.fit(X_train, y_train)
joblib.dump(selector, svm_model_baseDir+'lab1_fold5_recall.pkl')


selector = RFECV(estimator, step=1, cv=10, scoring='f1')
selector = selector.fit(X_train, y_train)
joblib.dump(selector, svm_model_baseDir+'lab1_fold10_f1.pkl')

selector = RFECV(estimator, step=1, cv=10, scoring='accuracy')
selector = selector.fit(X_train, y_train)
joblib.dump(selector, svm_model_baseDir+'lab1_fold10_acc.pkl')

selector = RFECV(estimator, step=1, cv=10, scoring='precision')
selector = selector.fit(X_train, y_train)
joblib.dump(selector, svm_model_baseDir+'lab1_fold10_prec.pkl')

selector = RFECV(estimator, step=1, cv=10, scoring='recall')
selector = selector.fit(X_train, y_train)
joblib.dump(selector, svm_model_baseDir+'lab1_fold10_recall.pkl')