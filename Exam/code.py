# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:37:19 2020

@author: Exam ID 273
"""

import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV 

x_test = np.genfromtxt('X_test.csv', delimiter=',')
x_train = np.genfromtxt('X_train.csv', delimiter=',')
y_test = np.genfromtxt('y_test.csv', delimiter=',')
y_train = np.genfromtxt('y_train.csv', delimiter=',')

# =============================================================================
#  Data understanding and preprocessing : Frequencies of Classes
# =============================================================================
classes, class_freq = np.unique(y_train, return_counts=True)
class_frequencies = np.asarray((classes, class_freq))

print("# ====================================================================")
print(" Frequencies of Classes")
print("# ====================================================================")
      
print('Classes and their frequencies: ')
print(class_frequencies)


# =============================================================================
#  Principal component analysis
# =============================================================================

x = StandardScaler().fit_transform(x_train)
x_standard_test = StandardScaler().fit_transform(x_test)

pca = PCA()

pca.fit(x)

# =============================================================================
#  Eigenspectrum
# =============================================================================
variances = pca.explained_variance_ratio_

eigenvalues = pca.explained_variance_
eigenvector = [i+1 for i in range(16)]

print("# ====================================================================")
print(" Eigenspectrum")
print("# ====================================================================")
plt.ylabel('eigenvalues')
plt.xlabel('number of eigenvectors')
plt.title('Eigenspectrum')
plt.plot(eigenvector, eigenvalues)
plt.savefig('Eigenspectrum.png')
plt.show()


# =============================================================================
#  Number of Principal components for 90% variance
# =============================================================================
print("# ====================================================================")
print(" Number of Principal components for 90% variance")
print("# ====================================================================")
      
cum_variance=np.cumsum(np.round(variances, decimals=4)*100)
print("cumulative Variances")
print(cum_variance)


# =============================================================================
# Scatter plot for PCA
# =============================================================================

principal_components = pca.fit_transform(x)

pc_col = ["principal_component_"+str(i+1) for i in range(16)]


pc_df = pd.DataFrame(data = principal_components , 
        columns = pc_col)
pc_df['label'] = y_train


plt.figure(figsize=(15,15))
sns.scatterplot(
    x="principal_component_1", y="principal_component_2",
    hue="label",
    palette=sns.color_palette("hls", 5),
    data=pc_df,
    legend="full",
    alpha=0.8
)



# =============================================================================
#  Clustering
# =============================================================================

kmeans = KMeans(n_clusters=5)
kmeans.fit(x)

centroids = kmeans.cluster_centers_

centroid_principal_components = pca.transform(centroids)


pc_centroid_df = pd.DataFrame(data = centroid_principal_components , 
        columns = pc_col)

pc_centroid_df['-------------------------'] = ['Cluster Centers' for i in range(5)]
sns.scatterplot(
    x="principal_component_1", y="principal_component_2",
    data=pc_centroid_df,
    hue="-------------------------",
    legend='full',
    alpha=1,s=100
)
plt.title("Cluster Centers")
plt.savefig('Cluster Centers.png')

# =============================================================================
#  Classification
# =============================================================================

# =============================================================================
#  Multi-nominal Logistic Regression
# =============================================================================
lr_classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs'
                                   ,max_iter=500)
lr_classifier.fit(x_train,y_train)

lr_train_pred = lr_classifier.predict(x_train)
lr_test_pred = lr_classifier.predict(x_test)

print("# ====================================================================")
print("Multi-nominal Logistic Regression")
print("# ====================================================================")
      
print("test loss :",metrics.zero_one_loss(y_test, lr_test_pred))
print("train loss :",metrics.zero_one_loss(y_train, lr_train_pred))




# =============================================================================
#  Random Forest
# =============================================================================
n_trees = [50, 100, 200]
print("# ====================================================================")
print("Random Forest")
print("# ====================================================================")
for i in n_trees :
    rf_classifier = RandomForestClassifier(n_estimators=i)
    rf_classifier.fit(x_train, y_train)
    train_pred = rf_classifier.predict(x_train)
    test_pred = rf_classifier.predict(x_test)
    print("test loss for ", i, " trees:", metrics.zero_one_loss(y_test, test_pred))
    print("train loss for ", i, " trees:", metrics.zero_one_loss(y_train, train_pred))


# =============================================================================
#  k-nearest-neighbor
# =============================================================================

grid_params = {'n_neighbors': [i for i in range(3,60,2)]}  
  
grid = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5, n_jobs=-1) 

grid.fit(x,y_train)

knn_train_pred = grid.predict(x)
knn_test_pred = grid.predict(x_standard_test)
print("# ====================================================================")
print("k-nearest-neighbor")
print("# ====================================================================")
print("number of neighbors", grid.best_params_)

print("test loss :",metrics.zero_one_loss(y_test, knn_test_pred))
print("train loss :",metrics.zero_one_loss(y_train, knn_train_pred))




