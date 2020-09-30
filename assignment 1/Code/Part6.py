# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:09:21 2019

@author: Aashish Kumar pcr902
"""

import numpy as np
import matplotlib.pyplot as plt

expData = np.loadtxt("DanWood.dt").reshape(6, 2)

temp = np.array([[i] for i in expData[:,0] ])
one = np.ones([6,1])
Y = np.transpose(expData[:,1])
X = np.append(temp, one, axis=1)
XT = np.transpose(X)
inv_XdotXT = np.linalg.inv(np.dot(XT,X))
LR = np.dot(np.dot(inv_XdotXT,XT),Y)
print(LR)

Fx=[]
for i in expData[:,0]:
    Fx.append(i*LR[0] + LR[1])

fig, ax = plt.subplots()
ax.plot(expData[:,0], Fx,label="regression line")
ax.scatter(expData[:,0], expData[:,1], color='red', label="data points")
ax.legend(loc='upper left')
ax.set(xlabel='X', ylabel='Y',
       title='Linear Regression ')
fig.savefig('part6_1.png')


plt.show()

Var= np.var(Y)
MSE = np.mean(np.square(np.subtract(Y,Fx)))
print(MSE)
print(Var)
print(MSE/Var)

X3point = np.array([i**3 for i in expData[:,0] ])
X3values = np.array([[i**3] for i in expData[:,0] ])
X3 = np.append(X3values, one, axis=1)
XT3 = np.transpose(X3)
inv_XdotXT3 = np.linalg.inv(np.dot(XT3,X3))
LR3 = np.dot(np.dot(inv_XdotXT3,XT3),Y)
print(LR3)

Fx3=[]
for i in X3point:
    Fx3.append(i*LR3[0] + LR3[1])

fig1, ax1 = plt.subplots()
ax1.plot(X3point, Fx3,label="regression line")
ax1.scatter(X3point, expData[:,1], color='red', label="data points")
ax1.legend(loc='upper left')
ax1.set(xlabel='X^3', ylabel='Y',
       title='Transformed Linear Regression X^3')
fig1.savefig('part6_2.png')
transformedMSE = np.mean(np.square(np.subtract(Y,Fx3)))
print(transformedMSE)
plt.show()
