# -*- coding: utf-8 -*-
"""
"""

#removing sklearn wornings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#import packages

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier



#data read
data = pd.read_csv('goal.csv')

#print(data.describe())

#print details about attribues
print("----------------------------------------------\n")
print(data.info())
print("----------------------------------------------\n")

#print correlation between attributes
print("----------------------------------------------\n")
print(data.corr(method='kendall'))
print("----------------------------------------------\n")

#removing 'Name' attribute
data.drop('Name',axis=1, inplace=True)

#removing 'Age' attribute
#data.drop('Age',axis=1, inplace=True)

#removing 'Venue' attribute
data.drop('Venue',axis=1, inplace=True)

#Target in numpy array
Y = data['Target'].values

#removing 'Target' attribute
data.drop('Target',axis=1, inplace=True)

#features in numpy array
X = data.values

#creating objects of the classifier
rfc = RandomForestClassifier()

dtc = DecisionTreeClassifier(random_state=0)

knn = KNeighborsClassifier(n_neighbors=5)

abc = AdaBoostClassifier(n_estimators=100)

naive = GaussianNB()

#declaring lists for Random Forest Classifier
accuracy_rfc = []
precision_rfc = []
recall_rfc = []
F1_rfc = []

#declaring lists for Decision Tree Classifier
accuracy_dtc = []
precision_dtc = []
recall_dtc = []
F1_dtc = []

#declaring lists  for K Nearest Neighbour Classifier
accuracy_knn = []
precision_knn = []
recall_knn = []
F1_knn = []

#declaring lists for adaBoost Classifier
accuracy_abc = []
precision_abc = []
recall_abc = []
F1_abc = []

#declaring lists for Gaussian Naive Bayes Classifier
accuracy_naive = []
precision_naive = []
recall_naive = []
F1_naive = []

#set value = 5 for 5 fold cross validation
val = 5

#K Fold Cross validation
kf = KFold(n_splits=val,random_state = 0,shuffle = True)

#call loop 5 times for 5 fold cross validation
for train_index, test_index in kf.split(X):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    #Scaling features and type casting 
    scaler = preprocessing.StandardScaler().fit(X_train.astype(float))
    X_train = scaler.transform(X_train.astype(float))
    X_test = scaler.transform(X_test.astype(float))
    
    
    #Model Traning, Prediction, Calculation for Random Forest 
    rfc.fit(X_train, y_train)
    y_pred_rfc = rfc.predict(X_test)
    
    accuracy_rfc.append( metrics.accuracy_score(y_test, y_pred_rfc))
    precision_rfc.append(metrics.precision_score(y_test, y_pred_rfc,average='macro'))
    recall_rfc.append(metrics.recall_score(y_test, y_pred_rfc,average='macro'))
    F1_rfc.append(metrics.f1_score(y_test, y_pred_rfc,average='macro'))
    
    #Model Traning, Prediction, Calculation for Decision Tree
    dtc.fit(X_train, y_train)
    y_pred_dtc = dtc.predict(X_test)
    
    accuracy_dtc.append( metrics.accuracy_score(y_test, y_pred_dtc))
    precision_dtc.append(metrics.precision_score(y_test, y_pred_dtc,average='macro'))
    recall_dtc.append(metrics.recall_score(y_test, y_pred_dtc,average='macro'))
    F1_dtc.append(metrics.f1_score(y_test, y_pred_dtc,average='macro'))
    
    #Model Training, Prediciton, Calculation for KNN
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    
    accuracy_knn.append( metrics.accuracy_score(y_test, y_pred_knn))
    precision_knn.append(metrics.precision_score(y_test, y_pred_knn,average='macro'))
    recall_knn.append(metrics.recall_score(y_test, y_pred_knn,average='macro'))
    F1_knn.append(metrics.f1_score(y_test, y_pred_knn,average='macro'))
    
    #Model Training, Prediciton, Calculation for adaBoost
    abc.fit(X_train, y_train)
    y_pred_abc = abc.predict(X_test)
    
    accuracy_abc.append( metrics.accuracy_score(y_test, y_pred_abc))
    precision_abc.append(metrics.precision_score(y_test, y_pred_abc,average='macro'))
    recall_abc.append(metrics.recall_score(y_test, y_pred_abc,average='macro'))
    F1_abc.append(metrics.f1_score(y_test, y_pred_abc,average='macro'))
    
    
    #Model Training, Prediciton, Calculation for Gaussian Naive Bayes
    naive.fit(X_train, y_train)
    y_pred_naive = naive.predict(X_test)
    
    accuracy_naive.append( metrics.accuracy_score(y_test, y_pred_naive))
    precision_naive.append(metrics.precision_score(y_test, y_pred_naive,average='macro'))
    recall_naive.append(metrics.recall_score(y_test, y_pred_naive,average='macro'))
    F1_naive.append(metrics.f1_score(y_test, y_pred_naive,average='macro'))
    

#Print Performance metric
print("----------------------------------------------\n")
print ("\n\nAverage Accuracy of all the classifiers:\n")
print("----------------------------------------------\n")
print ("     Random Forest  : ",np.mean(accuracy_rfc))
print ("     Decision Tree  : ",np.mean(accuracy_dtc))
print ("K Nearest neighbor  : ",np.mean(accuracy_knn))
print ("adaBoost Classifier : ",np.mean(accuracy_abc))
print ("Gaussia Naive Bayes : ",np.mean(accuracy_naive))

print("----------------------------------------------\n")
print ("\n\nAverage Precision of all the classifiers:\n")
print("----------------------------------------------\n")
print ("     Random Forest  : ",np.mean(precision_rfc))
print ("     Decision Tree  : ",np.mean(precision_dtc))
print ("K Nearest neighbor  : ",np.mean(precision_knn))
print ("adaBoost Classifier : ",np.mean(precision_abc))
print ("Gaussia Naive Bayes : ",np.mean(precision_naive))

print("----------------------------------------------\n")
print ("\n\nAverage Recall of all the classifiers:\n")
print("----------------------------------------------\n")
print ("     Random Forest  : ",np.mean(recall_rfc))
print ("     Decision Tree  : ",np.mean(recall_dtc))
print ("K Nearest neighbor  : ",np.mean(recall_knn))
print ("adaBoost Classifier : ",np.mean(recall_abc))
print ("Gaussia Naive Bayes : ",np.mean(recall_naive))

print("----------------------------------------------\n")
print ("\n\nAverage F1-score of all the classifiers:\n")
print("----------------------------------------------\n")
print ("     Random Forest  : ",np.mean(F1_rfc))
print ("     Decision Tree  : ",np.mean(F1_dtc))
print ("K Nearest neighbor  : ",np.mean(F1_knn))
print ("adaBoost Classifier : ",np.mean(F1_abc))
print ("Gaussia Naive Bayes : ",np.mean(F1_naive))



