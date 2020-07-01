# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 01:12:48 2019

@author: asus
"""

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
%matplotlib inline 
import matplotlib.pyplot as plt

df = pd.read_excel('C:/Users/asus/Desktop/breast tissue/BreastTissue2.xlsx')
df.head()

df.dtypes

feature_df = df[['I0', 'PA500', 'HFS', 'DA', 'Area', 'A/DA', 'Max IP','DR', 'P']]
X = np.asarray(feature_df)


df['Class'] = df['Class'].astype('int')
y = np.asarray(df['Class'])


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train) 

yhat = clf.predict(X_test)
yhat


from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,2,3,4,5,6])
np.set_printoptions(precision=6)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['car(1)','fad(2)','mas(3)','gla(4)','con(5)','adi(6)'],normalize= False,  title='Confusion matrix')

from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted') 

from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)

from sklearn.model_selection import cross_val_score
#Estimator'ımıza oluşturduğumuz modeli gönderiyoruz, cv : kaç parçaya bölüneceğini belirler.Genellikle 10 alınır.
accuracies = cross_val_score(estimator = clf, X = X_train, y = y_train, cv = 10)
print("Ortalama değer (mean): %",accuracies.mean()*100)
print("std: %",accuracies.std()*100)
