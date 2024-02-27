# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:18:25 2024

@author: user
"""


'''
Business Objectives:
    Develop a predictive model to forecast a company's 
    revenue or profit based on its characteristics.

Data Dictionary:
        
1)Company_ID: Unique identifier for each company (integer)
2)Industry: Industry in which the company operates (string)
3)Revenue: Annual revenue of the company (in USD) (float)
4)Employees: Number of employees in the company (integer)
5)Location: Location of the company's headquarters (string)
6)Profit: Annual profit of the company (in USD) (float)
7)Market_Share: Market share of the company within its industry (float)
8)Customer_Satisfaction: Customer satisfaction rating of the company (on a scale of 1 to 5) (integer)
9)Year_Founded: Year the company was founded (integer)
10)CEO: Name of the company's CEO (string)
11)Public: Binary variable indicating whether the company is publicly traded or not (0: No, 1: Yes) (integer)'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#EDA
# Load the dataset
df = pd.read_csv('Company_Data.csv')

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
from sklearn.datasets import load_Company_Data
company_data= load_Company_Data()
dir(company_data)

df=pd.DataFrame(company_data.data)
df.head()
df['target']=company_data.target
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
