# standard libraries for data annalysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import sklearn

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from collections import Counter

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import model_selection 
from sklearn import feature_selection
from sklearn import metrics
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score



data = pd.read_csv('/home/fridah/Downloads/Customer_Churn.csv')
# print(data.head(10))
# print(data.describe())
# print(data.dtypes)
# print(data.columns)
# print(data.shape)
# checking for any missing columns or values
# print(data.columns.to_series().groupby(data.dtypes).groups)
# print(data.isna().sum())
# lets see the number of customers that churned and thos that did not
# print(data['Churn'].value_counts())
# the percentage of customers that left and remained
retained= data[data.Churn =='No']
Churned= data[data.Churn =='Yes']
num_retained= retained.shape[0]
num_Churned= Churned.shape[0]
# The percentage of customers the churned and did not
# print(num_retained / (num_retained + num_Churned) * 100, "% of customers stayed with the company")
# print(num_Churned / (num_retained + num_Churned) * 100, "% of cutomers that left the company")

# identify the categorical data in your data set
data.select_dtypes('object').head()
# print(list(data.select_dtypes('object').columns.drop('customerID')))

# identify numerical data in your data set
data.select_dtypes('number').head()
# print(list(data.select_dtypes('number')))

# convert categorical data or non numeric data into numeric data
cols= ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']
le= LabelEncoder()
# print('>>>>>>>>>>>>>>>>>>>>>')

# x_names = set(data['PaymentMethod'])
# print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>names')
# print(x_names)
# my_list = {}
# for names in x_names:
#     counter_t = len(data['PaymentMethod'][data['PaymentMethod'] == names])
#     my_list[names] = counter_t
# sorted_list = {k:v for k,v in sorted(my_list.items(),key = lambda item : item[1])}
# print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>sorted list')
# print(sorted_list)
# data[cols] = data[cols].apply(LabelEncoder().fit_transform)
# # print(data.head())

# # lets visualize the data

# PaymentMethod= data.groupby('PaymentMethod').count()
# x = np.arange(len(x_names))

# plt.bar(list(sorted_list.keys()),list(sorted_list.values()))
# plt.xlabel('PaymentMethod', fontsize=18)
# plt.ylabel('No. Customers', fontsize=16)
# plt.title('PaymentMethod Distribution')
# plt.show()

# >>>>>>>>>>>>>>>>>>

# x_names = set(data['InternetService'])
# x = np.arange(len(x_names))
# InternetService= data.groupby('InternetService').count()
# my_list = {}
# for names in x_names:
#     counter_t = len(data['InternetService'][data['InternetService'] == names])
#     my_list[names] = counter_t
# plt.bar(list(my_list.keys()),list(my_list.values()))
# plt.xlabel('InternetService', fontsize=18)
# plt.ylabel('No. Customers', fontsize=16)
# plt.title('InternetService Distribution')
# plt.show()

# x_names = set(data['Contract'])
# x = np.arange(len(x_names))
# Contract= data.groupby('Contract').count()
# my_list = {}
# for names in x_names:
#     counter_t = len(data['Contract'][data['Contract'] == names])
#     my_list[names]= counter_t
# plt.bar(list(my_list.keys()),list(my_list.values()))
# plt.xlabel('Contract', fontsize=18)
# plt.ylabel('No. Customers', fontsize=16)
# plt.title('Contract Distribution')
# plt.show()

# x_names = set(data['Churn'])
# x = np.arange(len(x_names))
# Contract= data.groupby('Churn').count()
# my_list = {}
# for names in x_names:
#     counter_t = len(data['Churn'][data['Churn'] == names])
#     my_list[names]= counter_t
# plt.bar(list(my_list.keys()),list(my_list.values()))
# plt.xlabel('Churn', fontsize=18)
# plt.ylabel('% of Customers', fontsize=16)
# plt.title('Overall Churn Rate')
# plt.show()



identity= data["customerID"]
print(data.drop(columns="customerID"))
# convert the rest of the categorical data into dummy
data= pd.get_dummies(data)
print(pd.concat([data, identity], axis= 1))

#we want to scale down our data
X= data.drop(columns= ['Churn_No', 'Churn_Yes'])
y= data['Churn_Yes']
X= StandardScaler().fit_transform(X)
#split the data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 0)
print("Number transactions X_train data:", X_train.shape)
print("Number transactions y_train data:", y_train.shape)
print("Number transactions X_test data:", X_test.shape)
print("Number transactions y_test data:", y_test.shape)

#(Creating Logistic Regression
classifier= LogisticRegression(random_state= 0, penalty= 'l2')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
df= pd.DataFrame({'Accuracy':[metrics.accuracy_score(y_test, y_pred)],  
    'Precision':[metrics.precision_score(y_test, y_pred)], 'Recall':[metrics.recall_score(y_test, y_pred)], 
    'F1_score':[metrics.recall_score(y_test, y_pred)], 'F2_score':[metrics.fbeta_score(y_test, y_pred, beta=2.0)]},
    index=['LogisticRegression'])

print(df)

#Random Forest
classifier= RandomForestClassifier(n_estimators= 72, criterion= 'entropy', random_state= 0)
classifier.fit(X_train, y_train)
y_pred= classifier.predict(X_test)
df= pd.DataFrame({'Accuracy':[metrics.accuracy_score(y_test, y_pred)],  
    'Precision':[metrics.precision_score(y_test, y_pred)], 'Recall':[metrics.recall_score(y_test, y_pred)], 
    'F1_score':[metrics.recall_score(y_test, y_pred)], 'F2_score':[metrics.fbeta_score(y_test, y_pred, beta=2.0)]},
    index=['Random Forest'])

print(df)

# # Decision Tree
classifier = DecisionTreeClassifier(criterion= 'entropy', random_state= 0)
classifier.fit(X_train, y_train)
y_pred= classifier.predict(X_test)
df= pd.DataFrame({'Accuracy':[metrics.accuracy_score(y_test, y_pred)],  
    'Precision':[metrics.precision_score(y_test, y_pred)], 'Recall':[metrics.recall_score(y_test, y_pred)], 
    'F1_score':[metrics.recall_score(y_test, y_pred)], 'F2_score':[metrics.fbeta_score(y_test, y_pred, beta=2.0)]},
    index=['Decision Tree'])

print(df)

# /# Support Vector
# classifier = svm.SVC(kernel = 'rbf', random_state= 0)
# classifier.fit(X_train, y_train)
# y_pred= classifier.predict(X_test)
# print('Accuracy:',metrics.accuracy_score(y_test, y_pred))
# print('Precision:',metrics.precision_score(y_test, y_pred))
# print('Recall:',metrics.recall_score(y_test, y_pred))
# print('f1:',metrics.f1_score(y_test, y_pred))
# print('f2:',metrics.fbeta_score(y_test, y_pred, beta=2.0))

# # evaluating each model using k-fold cross validation
# kfold = model_selection.KFold(n_splits= 10, random_state= None)
# accuracies= cross_val_score
# #accuracy scoring:
# cv_acc_results = model_selection.cross_val_score(models, X_train, y_train, scoring= 'accuracy')
# # roc_auc scoring
# cv_auc_results = model_selection.cross_val_score(models, X_train, y_train, scoring= 'roc_auc')                 





