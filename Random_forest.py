# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:15:40 2024

@author: user
"""

import pandas as pd
from sklearn.datasets import load_digits
digits=load_digits()
dir(digits)
'''o/p: ['DESCR', 'data', 'feature_names', 'frame', 'images', 'target', 'target_names']'''
df=pd.DataFrame(digits.data)
df.head()
df['target']=digits.target
df[0:12]

X = df.drop('target',axis='columns')
Y=df.target

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=20)

#n_estimator:number of trees in the forest
model.fit(X_train,Y_train)

model.score(X_test ,Y_test)
y_predicted=model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_predicted)
cm
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')