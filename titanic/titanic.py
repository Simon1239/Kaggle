# -*- coding: utf-8 -*-
# Author   : ZhangQing
# Time     : 2022/12/7 17:07
# File     : titanic.py
# Project  : titanic
# Desc     :

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
import pandas as pd
from sklearn import linear_model

train = pd.read_csv('D:\\Practice\\Kaggle\\titanic\\data\\train.csv')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
# 设置1000列的时候才换行
pd.set_option('display.width', 1000)

def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进RandomForgestRegressor
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    y = known_age[:,0]
    x = known_age[:,1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x, y)

    # 用得到的模型来进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:,1::])
    df.loc[(df.Age.isnull()),'Age'] = predictedAges

    return df


def set_Cabin_type(df):

    df.loc[(df.Cabin.notnull()),'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()),'Cabin'] = 'No'
    return df


train = set_missing_ages(train)
train_over = set_Cabin_type(train)
dummies_Cabin = pd.get_dummies(train_over['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(train_over['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(train_over['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(train_over['Pclass'], prefix= 'Pclass')

df = pd.concat([train_over, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

scaler = preprocessing.StandardScaler()
df['Age_scaled'] = scaler.fit_transform(df[['Age']])
df['Fare_scaled'] = scaler.fit_transform(df[['Fare']])


# 用正则取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6,solver='liblinear')
clf.fit(X, y)

# print(clf)
# 测试数据处理

test = pd.read_csv('D:\\Practice\\Kaggle\\titanic\\data\\test.csv')
test.loc[ (test.Fare.isnull()), 'Fare'] = 0
test = set_missing_ages(test)
test_over = set_Cabin_type(test)

dummies_Cabin = pd.get_dummies(test_over['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(test_over['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(test_over['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(test_over['Pclass'], prefix= 'Pclass')
df_test = pd.concat([test_over, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

scaler = preprocessing.StandardScaler()
df_test['Age_scaled'] = scaler.fit_transform(df_test[['Age']])
df_test['Fare_scaled'] = scaler.fit_transform(df_test[['Fare']])

# 预测结果
test_end = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test_end)
result = pd.DataFrame({'PassengerId':test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("D:\\Practice\\Kaggle\\titanic\\data\\logistic_regression_predictions.csv", index=False)
print('预测完成')