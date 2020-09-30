# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 00:52:56 2020

@author: Aashish Kumar
"""
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

trainData = np.loadtxt("parkinsonsTrainStatML.dt")
testData = np.loadtxt("parkinsonsTestStatML.dt")


[featureTrainData, labelTrainData] = [trainData[:,:22], trainData[:,22]]
[featureTestData, labelTestData] = [testData[:,:22], testData[:,22]]


#Create a Gaussian Classifier
classifier = RandomForestClassifier(n_estimators=50)

#Train the model
classifier.fit(featureTrainData, labelTrainData)

test_pred = classifier.predict(featureTestData)

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(labelTestData, test_pred))
