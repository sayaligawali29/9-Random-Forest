# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:11:12 2024

@author: user
"""
'''
Business Objectives:
    The business objective is to predict the likelihood of
    diabetes in individuals based on their medical and 
    demographic information.
    
Data Dictionary:
    
1)Pregnancies: Number of times pregnant (integer)
2)Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test (integer)
3)BloodPressure: Diastolic blood pressure (mm Hg) (integer)
4)SkinThickness: Triceps skin fold thickness (mm) (integer)
5)Insulin: 2-Hour serum insulin (mu U/ml) (integer)
6)BMI: Body mass index (weight in kg/(height in m)^2) (float)
7)DiabetesPedigreeFunction: Diabetes pedigree function (a function that scores likelihood of diabetes based on family history) (float)
8)Age: Age in years (integer)
9)Outcome: Class variable (0: No diabetes, 1: Diabetes) (integer)
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#EDA
# Load the dataset
df = pd.read_csv('HR_DT.csv')

# Check for missing values
print(df.isnull().sum())

# Explore basic statistics
print(df.describe())

# Visualize the distribution of the target variable
sns.countplot(x='Outcome', data=df)
plt.show()

#Data Cleaning
# Handle missing values (if any)
df = df.dropna()  # Remove rows with missing values

# Check for outliers and decide on a strategy (e.g., removing or transforming)
# For simplicity, let's remove outliers from numerical features
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
df = df[(df[numeric_features] < df[numeric_features].quantile(0.95)).all(axis=1)]

# Encode categorical variables (if any)
# Assuming there are no categorical variables in this dataset

# Check the cleaned dataset
print(df.head())
#Random Forest ALgorithm
import pandas as pd
from sklearn.datasets import load_HR_DT
hr_dt= load_HR_DT()
dir(hr_dt)

df=pd.DataFrame(hr_dt.data)
df.head()
df['target']=hr_dt.target
df[0:12]

X = df.drop('target',axis='columns')
Y=df.target

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=40)

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
