#=====================================================#
# STUDENT SCORE PREDICTION MODEL
#=====================================================#

#----------------IMPORT LIBRARIES---------------------#

from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#--------------------LOAD DATASET----------------------#
data=pd.read_csv("StudentPerformanceFactors.csv") 
print("shape=",data.shape)
 
#-------------------BASIC DATA CHECK-------------------# 
print('\nINFO')
print(data.info())

print("\nMISSING VALUES")
print(data.isnull().sum())

print("\nDESCRIPTION")
print((data.describe))

#------------------TARGET DISTRIBUTION-----------------#
plt.hist(data['Exam_Score'],bins=50)
plt.title('Target Distribution')
plt.xlabel('Exam Score')
plt.ylabel('Count')
plt.show()

#--------------ENCODE CATAGORICAL DATA-----------------#
data_encoded=pd.get_dummies(data,drop_first=True,dtype=int)

#-----------------CORRELATION HEATMAP-------------------#
plt.figure(figsize=(10,6))
sns.heatmap(data_encoded.corr(), cmap="coolwarm")
plt.title('Correlation Matrix')
plt.show()

#-------------OUTLIER REMOVER (IQR METHOD)-------------#
Q1=data_encoded.quantile(0.25)
Q3=data_encoded.quantile(0.75)
IQR=Q3-Q1
data_clean=data_encoded[~((data_encoded < (Q1-1.5*IQR))   |
                          (data_encoded > (Q3+1.5*IQR))).any(axis=1)]
print('After outlier remove', data_clean.shape)

#-----------------FEATURE ENGINEERING------------------#
if "Hours_Studied" in data_clean.columns and "Sleep_Hours" in data_clean.columns:
    data_clean['Study_Sleep_Ratio']=data_clean['Hours_Studied']/(data_clean['Sleep_Hours']+1)

#-----------------FEATURES AND TARGET------------------#
x=data_clean.drop("Exam_Score",axis=1)
y=data_clean['Exam_Score']

#-----------------TRAIN TEST SPLIT---------------------#
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=42)

#-----------------------SCALING------------------------#
st=StandardScaler()
x_train_scaled=st.fit_transform(x_train)
x_test_scaled=st.transform(x_test)

#======================================================#
#             MODEL 1 - LINEAR REGRESSION
#======================================================#
lr=LinearRegression()
lr.fit(x_train_scaled,y_train)
lr_pred=lr.predict(x_test_scaled)
print("\nLINEAR REGRESSION")
print("MSE",mean_squared_error(y_test,lr_pred))
print("R2",r2_score(y_test,lr_pred))
# OVERFITTING CHECK
train_pred_lr=lr.predict(x_train_scaled)
print('Train R2',r2_score(y_train,train_pred_lr))
print('Test R2',r2_score(y_test,lr_pred))

#======================================================#
#                   MODEL 2 - RIDGE
#======================================================#
Ridge=Ridge(alpha=1.0)
Ridge.fit(x_train_scaled,y_train)
Ridge_pred=Ridge.predict(x_test_scaled)
print('\nRidge Regression')
print('R2',r2_score(y_test,Ridge_pred))

#======================================================#
#              MODEL 3 - RANDOM FOREST
#======================================================#
rf=RandomForestRegressor(n_estimators=300,random_state=42)
rf.fit(x_train_scaled,y_train)
rf_pred=rf.predict(x_test_scaled)
print('\nRandom Forest')
print("MSE",mean_squared_error(y_test,rf_pred))
print("R2",r2_score(y_test,rf_pred))

#======================================================#
#                 MODEL COMPARISION
#======================================================#
print('\nModel Comparision')
print("R2",r2_score(y_test,lr_pred))
print('R2',r2_score(y_test,Ridge_pred))
print("R2",r2_score(y_test,rf_pred))

#======================================================#
#                FEATURE IMPORTANCE
#======================================================#
importance=pd.Series(lr.coef_,index=x.columns)
importance=importance.sort_values(ascending=False).head(10).plot(kind='bar')
plt.title('Top Important Features')
plt.show()

#======================================================#
#               ACTUAL VS PREDICTED
#======================================================#
plt.scatter(y_test,lr_pred)
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('Actual vs Predicted (LR)')
plt.show()

#======================================================#
#                  RESIDUAL PLOT
#======================================================#
Residual=y_test-lr_pred
plt.scatter(lr_pred,Residual)
plt.axhline(y=0)
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.title('Residual Plot')
plt.show()
