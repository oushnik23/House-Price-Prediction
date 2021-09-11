# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 11:00:26 2021

@author: Administrator
"""

import os
os.getcwd()
os.chdir("C:/Users/Administrator/Desktop/PYTHON/house pricing")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")

test_data.columns

#Generally researchers believe that a correlation coefficient of 0.3 or less is low correlation, 0.3 to 0.7 is medium correlation, and 0.7 or more is high correlation.
#Here we remove the features that are low related to the target

corref=train_data.corr()
imp=corref.index[abs(corref['SalePrice'])>.3]
imp

train2=train_data[['LotFrontage', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea','BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea','FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars','GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'SalePrice']]
test=test_data[['LotFrontage', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea','BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea','FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars','GarageArea', 'WoodDeckSF', 'OpenPorchSF']]

train2.dtypes

#Searching for numerical and catagorical feature 
numeric=(train2.dtypes=='object')
object_cols=list(numeric[numeric].index)
print(object_cols)

cat=(train2.dtypes!='object')
object_num=list(cat[cat].index)
print(object_num)

#missing value checking
test.isnull().sum().sort_values(ascending=False)
train2.isnull().sum().sort_values(ascending=False)

train_data['GarageYrBlt'].value_counts()
train2.LotFrontage.unique()

mean=train2['LotFrontage'].mean()
train2['LotFrontage']=train2['LotFrontage'].fillna(mean)
test['LotFrontage']=test['LotFrontage'].fillna(test['LotFrontage'].mean())

train2['GarageYrBlt']=train2['GarageYrBlt'].fillna(train2.GarageYrBlt.mean())
test['GarageYrBlt']=test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean())

train2['MasVnrArea']=train2['MasVnrArea'].fillna(train2.MasVnrArea.mean())

test['MasVnrArea']=test['MasVnrArea'].fillna(test.MasVnrArea.mean())
test['GarageArea']=test['GarageArea'].fillna(test.GarageArea.mean())
test['GarageCars']=test['GarageCars'].fillna(test.GarageCars.mean())
test['BsmtFinSF1']=test['BsmtFinSF1'].fillna(test.BsmtFinSF1.mean())
test['TotalBsmtSF']=test['TotalBsmtSF'].fillna(test.TotalBsmtSF.mean())



#train_data=train_data.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageCond','GarageFinish','GarageQual','BsmtExposure','BsmtFinType1','BsmtFinType1','BsmtFinType2','MasVnrType','MSZoning','GarageType','BsmtCond','LotFrontage','BsmtQual','MasVnrArea','Electrical'],axis=1)
#test_data=test_data.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageCond','GarageFinish','GarageQual','BsmtExposure','BsmtFinType1','BsmtFinType1','BsmtFinType2','MasVnrType','MSZoning','GarageType','BsmtCond','LotFrontage','BsmtQual','MasVnrArea','Electrical'],axis=1)

x=train2.drop(['SalePrice'],axis=1)
y=train2[['SalePrice']]

#x=pd.get_dummies(x)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.2)

#Linear regression
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)

y_pred=model.predict(x)

y_pred2=model.predict(xtest)


from sklearn.metrics import r2_score 
r2_score(ytest,y_pred2)

#Decision tree
from sklearn.tree import DecisionTreeRegressor
dt_model=DecisionTreeRegressor()
dt_model.fit(x,y)

y_pred2=dt_model.predict(xtest)

r2_score(ytest,y_pred2)

prediction2=dt_model.predict(test)

predict=pd.DataFrame({'ID':test_data.Id, 'SalePrice':prediction2}) 

predict.to_csv('my_submision.csv', index=False)

