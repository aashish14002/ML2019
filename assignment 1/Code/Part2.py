# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:12:47 2019

@author: Aashish Kumar pcr902
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

eventOutcome = [0, 1]
empiricalFrequency = []
for i in range(1000000):
    count = 0
    for j in range(20):
        if(random.choice(eventOutcome)):
            count += 1
    count /= 20
    empiricalFrequency.append(count)
    
alpha = np.arange(0.5, 1.05, 0.05)
alpha_count = []
makrov_bound = []
chebshev_bound = []
empirical = np.asarray(empiricalFrequency)

for i in range(alpha.size):
    alpha_count.append(np.count_nonzero(empiricalFrequency >= alpha[i])/1000000.0)
    makrov_bound.append(0.5/alpha[i])
    if(alpha[i]!=0.5):
        cheb_bound = 0.0125/((alpha[i]-0.5)**2)
    else:
        cheb_bound = 1
    if(cheb_bound > 1.0):
        chebshev_bound.append(1.0)
    else:
        chebshev_bound.append(cheb_bound)
    

print(alpha)
    
fig, ax = plt.subplots()
ax.plot(alpha, alpha_count,label="empirical frequency")
ax.plot(alpha, makrov_bound, label="makrov bound")
ax.plot(alpha, chebshev_bound, label="chebyshev bound")
ax.legend(loc='upper right')

ax.set(xlabel='alpha', ylabel='concentration measure',
       title='Concentration Measures')

fig.savefig('Part2.png')

plt.show()