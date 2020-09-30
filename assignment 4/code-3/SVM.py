# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 02:07:27 2019

@author: Aashish Kumar pcr902
"""
import pandas as pd 
import numpy as np 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC
import matplotlib.pyplot as plt

trainData = np.loadtxt("parkinsonsTrainStatML.dt")
testData = np.loadtxt("parkinsonsTestStatML.dt")


[featureTrainData, labelTrainData] = [trainData[:,:22], trainData[:,22]]
[featureTestData, labelTestData] = [testData[:,:22], testData[:,22]]


##########################################################################
# Part 1 Data Normalization
#
##########################################################################

trainDataMean = np.mean(featureTrainData, axis=0)
trainDataVar = np.var(featureTrainData, axis=0)
trainDatastd = np.sqrt(trainDataVar)

print("Mean of training Data set:")
print(trainDataMean)

print(" Variance of training Data set:")
print(trainDataVar)

normalizedFeatureTrainData = (featureTrainData - trainDataMean)/trainDatastd # f(x)=(x-mean)/std 

normalizedTrainDataMean = np.mean(normalizedFeatureTrainData, axis=0)
normalizedTrainDataVar = np.var(normalizedFeatureTrainData, axis=0)

print("Mean of transformed train Data set:")
print(normalizedTrainDataMean)

print(" Variance of transformed train Data set:")
print(normalizedTrainDataVar)

normalizedFeatureTestData = (featureTestData - trainDataMean)/trainDatastd

normalizedTestDataMean = np.mean(normalizedFeatureTestData, axis=0)
normalizedTestDataVar = np.var(normalizedFeatureTestData, axis=0)

print("Mean of transformed test Data set:")
print(normalizedTestDataMean)

print(" Variance of transformed test Data set:")
print(normalizedTestDataVar)

##########################################################################
# Model selection using Grid Search
#
##########################################################################


paramGrid = {'C': [7,8,9,10,11,12,13],  
              'gamma': [ 0.001, 0.01, 0.1, 1, 10, 100,1000], 
              'kernel': ['rbf']}  
  
grid = GridSearchCV(estimator=SVC(),cv=5 ,param_grid=paramGrid) 
  

grid.fit(normalizedFeatureTrainData, labelTrainData)

print("best parameters ")
print(grid.best_params_) 
  



gridPredictionTrain = grid.predict(normalizedFeatureTrainData) 
gridPredictionTest = grid.predict(normalizedFeatureTestData) 

lossTrain = np.sum(np.array([1 for i in range(gridPredictionTrain.size) if gridPredictionTrain[i] != labelTrainData[i]]))/gridPredictionTrain.size
lossTest = np.sum(np.array([1 for i in range(gridPredictionTest.size) if gridPredictionTest[i] != labelTestData[i]]))/gridPredictionTest.size

print("training and test data loss ")
print(lossTrain)
print(lossTest)



##########################################################################
# For calculating the bound in Airline Question
#
##########################################################################
p=1
for i in range(100):
    p*= (9600-i)/(10100-i)
print("probability: ")    
print(p)









