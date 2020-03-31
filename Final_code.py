# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 22:44:40 2020

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 17:05:06 2020

@author: HP
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


dataset=pd.read_csv("resources/loan_train.csv")
print(dataset.head(5))
dataset.columns

print(dataset.shape)

dataset.describe
dataset_missing=dataset.columns[dataset.isnull().any()].tolist()

print(dataset_missing)
cols=dataset[['interest_rate',
       'unpaid_principal_bal', 'loan_term', 
       'loan_to_value', 'number_of_borrowers',
       'debt_to_income_ratio', 'borrower_credit_score', 
       'insurance_percent', 'co-borrower_credit_score', 'insurance_type', 'm1',
       'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12']]#input
X=cols
Y=dataset.m13

#SPLITING THE DATA INTO TRAIN AND TEST
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.25,random_state=0)

#SCALING THE DATA USING MIN-MAX SCALER
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
numerical=['interest_rate',
       'unpaid_principal_bal', 'loan_term',
       'loan_to_value', 'number_of_borrowers',
       'debt_to_income_ratio', 'borrower_credit_score',
       'insurance_percent', 'co-borrower_credit_score', 'm1',
       'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12']
features_minmax_transform=pd.DataFrame(data=X)
features_minmax_transform[numerical]=scaler.fit_transform(X[numerical])
print(features_minmax_transform)

#TRAINING LOGISTIC REGRESSION MODEL
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(xtrain,ytrain)
print(xtrain)
y_pred=logreg.predict(xtest)

#CONFUSION MATRIX
from sklearn import metrics
confusion_matrix=metrics.confusion_matrix(ytest,y_pred)
print(confusion_matrix)

#VISUALISATION
class_names=[0,1] #name of the classes
fig,ax=plt.subplots()
tick_set=np.arange(len(class_names))
plt.xticks(tick_set,class_names)
plt.yticks(tick_set,class_names)

#CREATE A HEATMAP
sns.heatmap(pd.DataFrame(confusion_matrix),annot=True,cbar=True,cmap='YlGnBu',fmt='g')
ax.xaxis.set_label_position("bottom")
plt.tight_layout()
plt.title("Exampleset")
plt.ylabel("Actual label")
plt.xlabel("predicted label")

#CONFUSION MATRIX EVALUATION METRICS
print("Accuracy:",metrics.accuracy_score(ytest, y_pred))

#ROC CURVE
y_pred_proba = logreg.predict_proba(xtest)[::,1]
fpr, tpr, _ = metrics.roc_curve(ytest,  y_pred_proba)
auc = metrics.roc_auc_score(ytest, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

#DECISION TREE MODEL BUILDING:
from sklearn.tree import DecisionTreeClassifier
dct=DecisionTreeClassifier()
dct.fit(xtrain,ytrain)
tree_pred=dct.predict(xtest)
print("Accuracy_decision_tree:",metrics.accuracy_score(ytest, tree_pred))

#VISUALISATION OF DECISION TREE

#IMPORT RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(xtrain,ytrain)
rfc_pred=rfc.predict(xtest)
print("Accuracy_RFC:",metrics.accuracy_score(ytest, rfc_pred))






