import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools # advanced tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

data=  pd.read_csv("/home/fridah/Downloads/creditcard.csv")
print(data.head(5))
# print(data.shape)
# print(data.dtypes)
# print(data.describe())
# print(data.dtypes.value_counts())
# print(data.isnull().sum)
# print(data.columns)
# print(data["Amount"].describe())

# # Get the no of genuine and fraud transactions
# non_fraud= len(data[data.Class == 0])
# fraud= len(data[data.Class == 1])
# fraud_percent= (fraud / (fraud + non_fraud)) * 100
# print("Number of Genuine transactions:", non_fraud)
# print("Number of Fraud transaction:", fraud)
# print("Percentage of Fraud transactions: {:.4f}".format(fraud_percent))

#Amount details of the fraud transaction
# print("Amount details of Fraudulent Transactions")
# print(fraud.Amount.describe())
# print("Amount details of Genuine Transactions")
# print(non_fraud.Amount.describe())

# lets visualize
# Labels= ["Genuine", "Fraud"]
# count_classes= data.value_counts(data['Class'], sort= True)
# count_classes.plot(kind= 'bar', rot= 0)
# plt.title("Visualization of Labels")
# plt.ylabel("Count")
# plt.xticks(range(2), Labels)
# plt.show()

# Building a correlation matrix
# corrMatrix = data.corr()
# fig=plt.figure(figsize= (12, 12))
# sns.heatmap(corrMatrix, vmax= .8, square= True)
# plt.show() SVM
# svm= SVC()
# svm.fit(X_train, y_train)
# svm_model= svm.predict(X_test)

# # XGBOOST
# xgb= XGBClassifier(max_depth= 4)
# xgb.fit(X_train, y_train)
# xgb_model= xgb.predict(X_test)

# # LoGISTIC REGRESSION
# lr= LogisticRegression()
# lr.fit(X_train, y_train)
# lr_model= lr.predict(X_test)

# Scale down our data
scaler= StandardScaler()
amount= data['Amount'].values
data['Amount']= scaler.fit_transform(amount.reshape(-1, 1))
print(data['Amount'].head(10))

# Feature selection and splitting
X= data.drop(["Class"], axis= 1).values
y= data["Class"].values
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state= 0)
print("shape of X_train", X_train[:1])
print("shape of X_test", X_test[0:1])
print("shape of y_train", y_train[0:20])
print("shape of y_test", y_test[0:20])

# Apply the aligorithm
# DECISION TREE
decision_tree= DecisionTreeClassifier(max_depth= 4, criterion= 'entropy')
decision_tree.fit(X_train, y_train)
decision_model= decision_tree.predict(X_test)
print("Accuracy score of the Decision Tree model is{}".format(accuracy_score(y_test, decision_model)))
print("F1 score of the Decision Tree model is{}".format(f1_score(y_test, decision_model)))

# KNN
n= 5
knn= KNeighborsClassifier(n_neighbors= n)
knn.fit(X_train, y_train)
knn_model= knn.predict(X_test)
print("Accuracy score of the KNN model is{}".format(accuracy_score(y_test, knn_model)))
print("F1 score of the KNN model is{}".format(f1_score(y_test, knn_model)))

# # LoGISTIC REGRESSION
lr= LogisticRegression()
lr.fit(X_train, y_train)
lr_model= lr.predict(X_test)
print("Accuracy score of the Logistic Regression model is{}".format(accuracy_score(y_test, lr_model)))
print("F1 score of the Logistic Regression model is{}".format(f1_score(y_test, lr_model)))

# # SVM
svm= SVC()
svm.fit(X_train, y_train)
svm_model= svm.predict(X_test)
print("Accuracy score of the SVM model is{}".format(accuracy_score(y_test, svm_model)))
print("F1 score of the SVM model is{}".format(f1_score(y_test, svm_model)))

# RANDOM FOREST
rf= RandomForestClassifier(max_depth= 4)
rf.fit(X_train, y_train)
rf_model= rf.predict(X_test)
print("Accuracy score of the Random Forest Tree model is{}".format(accuracy_score(y_test, rf_model)))
print("F1 score of the Random Forest Tree model is{}".format(f1_score(y_test, rf_model)))


# # # XGBOOST
# xgb= XGBClassifier(max_depth= 4)
# xgb.fit(X_train, y_train)
# xgb_model= xgb.predict(X_test)
# print("Accuracy score of the XGBoost model is{}".format(accuracy_score(y_test, xgb_model)))

# Lets evaluate  the accuracyscore
# print("Accuracy_score")
# print("Accuracy score of the Decision Tree model is{}".format(accuracy_score(y_test, decision_model)))
# print("Accuracy score of the KNN model is{}".format(accuracy_score(y_test, knn_model)), color= 'green')
# print("Accuracy score of the Random Forest Tree model is{}".format(accuracy_score(y_test, rf_model)))
# print("Accuracy score of the SVM model is{}".format(accuracy_score(y_test, svm_model)))
# print("Accuracy score of the XGBoost model is{}".format(accuracy_score(y_test, xgb_model)), color= 'red')
# print("Accuracy score of the Logistic Regression model is{}".format(accuracy_score(y_test, lr_model)))

# print("f1_score")
# print("F1 score of the Decision Tree model is{}".format(f1_score(y_test, decision_model)))
# print("F1 score of the KNN model is{}".format(f1_score(y_test, knn_model)))
# print("F1 score of the Random Forest Tree model is{}".format(f1_score(y_test, rf_model)))
# print("F1 score of the SVM model is{}".format(f1_score(y_test, svm_model)))
# print("F1 score of the Logistic Regression model is{}".format(f1_score(y_test, lr_model)))


# CONFUSION MATRIX
# tree_matrix= confusion_matrix(y_test, decision_model, labels= [0, 1])
# knn_matrix= confusion_matrix(y_test, knn_model, labels= [0, 1])
# lr_matrix= confusion_matrix(y_test, lr_model, labels= [0, 1])
# svm_matrix= confusion_matrix(y_test, svm_model, labels= [0, 1])
# rf_matrix= confusion_matrix(y_test, rf_model, labels= [0, 1])
# xgb_matrix= confusion_matrix(y_test, xgb_model, labels= [0, 1])

confusion_matrix_dt = confusion_matrix(y_test, decision_model.round())
print("Confusion Matrix - Decision Tree")
print(confusion_matrix_dt)
plot_confusion_matrix(confusion_matrix_dt, classes=[0, 1], title= "Confusion Matrix - Decision Tree")

# def plot_confusion_matrix(cm, classes, title, normalize = False, cmap = plt.cm):
#     title= 'Confusion Matrix of{}'.format(title)
#     if normalize:
#         cm= cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
#     plt.imshow(cm, interpolation= 'nearest', cmap= 'Blues')
#     plt.title(title)
#     plt.colorbar()
#     tick_marks= np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation= 45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment = 'center',
#                  color = 'white' if cm[i, j] > thresh else 'black')
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
# # compute confusion matrix for the models
# tree_matrix= confusion_matrix(y_test, decision_model, labels= [0, 1])
# knn_matrix= confusion_matrix(y_test, knn_model, labels= [0, 1])
# lr_matrix= confusion_matrix(y_test, lr_model, labels= [0, 1])
# svm_matrix= confusion_matrix(y_test, svm_model, labels= [0, 1])
# rf_matrix= confusion_matrix(y_test, rf_model, labels= [0, 1])
# # plot the confusion matrix
# plt.rcParams["figure.figsize"]= (6,6)

# Decision Tree confusion matrix

# tree_cm_plot = plot_confusion_matrix(tree_matrix, 
#                                 classes = ['Non-Default(0)','Default(1)'], 
#                                 normalize = False, title = 'Decision Tree')
# plt.savefig('tree_cm_plot.png')
# plt.show()

# Knn confusion matrix
# knn_cm_plot = plot_confusion_matrix(knn_matrix, 
#                                 classes = ['Non-Default(0)','Default(1)'], 
#                                 normalize = False, title = 'KNN')
# plt.savefig('knn_cm_plot.png')
# plt.show()



# sns.heatmap(plot_confusion_matrix(tree_matrix, classes= ['Non-Default(0)', 'Default(1)'], normalize= False, title= 'Decision Tree'))
# plt.savefig('tree_cm_plot.png')
# plt.show()





# Tackle the issue of data imbalance
# X_resampled, Y_resampled= SMOTE().fit_resample(X, Y)
# print("Resampled shape of X:", X_resampled.shape)
# print("Resampled shape of Y:", Y_resampled.shape)














