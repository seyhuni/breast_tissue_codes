# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 00:53:44 2019

@author: asus
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline

df = pd.read_excel('BreastTissue2.xlsx')
df.head()

cdf = df[['Class','I0', 'PA500', 'HFS', 'DA', 'Area', 'A/DA', 'Max IP','DR', 'P']] 
cdf.head(9)

plt.scatter(cdf.I0, cdf.Class,  color='blue')
plt.xlabel("IO")
plt.ylabel("Class")
plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

plt.scatter(cdf.I0, cdf.Class,  color='blue')
plt.xlabel("IO")
plt.ylabel("Class")
plt.show()

from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['I0', 'PA500', 'HFS', 'DA', 'Area', 'A/DA', 'Max IP','DR', 'P']])
y = np.asanyarray(train[['Class']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)

y_hat= regr.predict(test[['I0', 'PA500', 'HFS', 'DA', 'Area', 'A/DA', 'Max IP','DR', 'P']])
x = np.asanyarray(test[['I0', 'PA500', 'HFS', 'DA', 'Area', 'A/DA', 'Max IP','DR', 'P']])
y = np.asanyarray(test[['Class']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))


