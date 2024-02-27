# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:08:50 2024

@author: user
"""

"""Business Objectives:
    The business objective could be to develop a
    predictive model to detect fraudulent taxable income
    claims based on customer demographic and financial
    attributes. 
    
Data Dictionary:
1)Undergrad: Whether the customer (Categorical: Yes/No)
2)Marital_Status: Marital status of the customer (Categorical: Single/Married/Divorced)
3)Taxable_Income: Taxable income of the customer(Continuous: Integer)
4)City_Population: Population of the city  (Continuous: Integer)
5)Work_Experience: Work experience of the custome (Continuous: Integer)
6)Urban: Whether the customer (Categorical: Yes/No)
7)Mortgage: Mortgage value in USD (Continuous: Integer)
8)House_Ownership: Type of house ownership (Categorical: Own/Rent)
9)Car_Ownership: Whether the customer owns a car or not (Categorical: Yes/No)
10)Income_Category: Income category of the customer (Categorical: Low/Medium/High)
11)Fraud_Taxable: Whether the customer's taxable income is fraudulent or not (Target Variable) (Categorical: Yes/No)"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#EDA
# Load the dataset
df = pd.read_csv('Fraud_check.csv')

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
#Random Forest Algorithm
import pandas as pd
from sklearn.datasets import load_Fraud_check
fraud_check= load_Fraud_check()
dir(fraud_check)

df=pd.DataFrame(fraud_check.data)
df.head()
df['target']=fraud_check.target
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
