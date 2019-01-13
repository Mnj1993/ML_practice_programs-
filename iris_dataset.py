# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 02:39:28 2019

@author: manoj007
"""

import sys
import pandas as pd
from sklearn.model_selection import train_test_split

#print(dataset.5.1)
#------data preprocessing----------
columnes=['A','B','C','D','Y']
dataset= pd.read_csv("iris.data.txt",names=columnes)
print(dataset.describe())
y=dataset.Y
features=['A','B','C','D']
x=dataset[features].astype(float)
print (x)
train_x,test_x,train_y,test_y = train_test_split(x,y, random_state=0)

#---------standardisation-----------
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
train_x_std=sc.fit_transform(train_x)
test_x_std=sc.fit_transform(test_x)


#-------------onehotencoder-------------
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelencodr=LabelEncoder()
tray=labelencodr.fit_transform(train_y).reshape(1,-1)
test=labelencodr.fit_transform(test_y).reshape(1,-1)
#--------SVM------------------------
from sklearn.svm import SVC
decis=SVC(kernel='rbf',random_state=0)
decis.fit(train_x,train_y)
y_svc=pd.DataFrame(decis.predict(test_x))
y_svc.describe()
print("svc score:{}".format(decis.score(test_x,test_y)))

#----------------------Xgboost--------------
import xgboost as XGB
XGB1=XGB.XGBClassifier()
XGB1.fit(train_x,train_y)
y_xgboost_prediction=pd.DataFrame(XGB1.predict(test_x))
print("xgboost score :{}",format(XGB1.score(test_x,test_y)))

#-----------RandomForest-------------
from sklearn.ensemble import RandomForestClassifier
deci=RandomForestClassifier()
deci.fit(train_x,train_y)
y_pred=list(deci.predict(test_x))
print("random forest score :{}",format(deci.score(test_x,test_y)))

#-----------------K-Nearest Neighbors----------------
from sklearn.cluster import KMeans
kmean = KMeans(n_clusters=2)
kmean.fit(train_x,train_y)
y_kmean_pred=list(kmean.predict(test_x))
print("kmean score:{}".format(kmean.score(test_x,test_y)))

#--------------Naive bayes---------------------------
from sklearn.naive_bayes import GaussianNB
naive=GaussianNB()
GaussianNB.fit(train_x,tray,test)
#-----------accuract_score----------------
from sklearn.metrics import accuracy_score
print(accuracy_score(test_y,y_pred))
