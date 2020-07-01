# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 00:24:46 2019

@author: asus
"""

import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

my_data = pd.read_excel('BreastTissue.xlsx')
my_data.head()

X = my_data[['I0', 'PA500', 'HFS', 'DA', 'Area', 'A/DA', 'Max IP','DR', 'P']] .values  #.astype(float)
X[0:5]

y = my_data['Class'].values
y[0:5]

from sklearn.model_selection import train_test_split

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

breasTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
breasTree # it shows the default parameters

breasTree.fit(X_trainset,y_trainset)

predTree = breasTree.predict(X_testset)

print (predTree [0:5])
print (y_testset [0:5])

from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
#!conda install -c conda-forge pydotplus -y
#!conda install -c conda-forge python-graphviz -y

from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
%matplotlib inline 
import os

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"
dot_data = StringIO()
filename = "breasTree.png"
featureNames = my_data.columns[2:11]
featureNames
targetNames = my_data["Class"].unique().tolist()
out=tree.export_graphviz(breasTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')


from sklearn.model_selection import cross_val_score
#Estimator'ımıza oluşturduğumuz modeli gönderiyoruz, cv : kaç parçaya bölüneceğini belirler.Genellikle 10 alınır.
accuracies = cross_val_score(estimator = breasTree, X = X_trainset, y = y_trainset, cv = 10)
print("Ortalama değer (mean): %",accuracies.mean()*100)
print("std: %",accuracies.std()*100)

from matplotlib.colors import ListedColormap
X_set, y_set = X_trainset, y_trainset
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, breasTree.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Yas')
plt.ylabel('Maas')
plt.legend()
plt.show()


