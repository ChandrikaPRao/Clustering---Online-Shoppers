# -*- coding: utf-8 -*-
"""
This program is to solve a clustering problem on Online Shoppers Intention 
dataset (provided at UCI Machine Learning Repository) to provide useful 
insights. It starts with preprocessing of data which includes encoding for 
the categorical fields,followed by implementation of KMeans and 
AgglomerativeClustering algorithms. Rand index and davies_bouldin_score is 
calculated for both the algorithms for comparison purposes.
"""

import numpy as np
import pandas as pd


#import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score
from sklearn.feature_selection import SelectKBest    
from sklearn.feature_selection import chi2
from sklearn.ensemble import IsolationForest
from scipy import sparse as sp
from scipy.special import comb
from itertools import combinations
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer

import numpy

#from matplotlib.mlab import PCA as mlabPCA
from matplotlib import pyplot as plt

np.random.seed(42)

#$#############################################################################
def rand_index_score(labels_true, labels_pred):
    """
    Implementation 1: The function will calcuate the rand index
    """
    tp_plus_fp = comb(np.bincount(labels_true), 2).sum()
    tp_plus_fn = comb(np.bincount(labels_pred), 2).sum()
    A = np.c_[(labels_true, labels_pred)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(labels_true))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn

    return (tp + tn) / (tp + fp + fn + tn)
#$#############################################################################
def calculate_rand_index(labels_true,labels_pred):
    """
    Implementation 2: The function will calcuate the rand index
    """
    true_comb = combinations(labels_true, 2) 
    print("Type of true_comb :",type(true_comb))
    pred_comb = combinations(labels_pred, 2) 

    S = 0
    D = 0
    n = len(labels_true) 
    for i, j in zip(list(true_comb), list(pred_comb)):
            if(i[0]==i[1] and j[0]==j[1]):
                S += 1
            elif(i[0]!=i[1] and j[0]!=j[1]):
                D += 1
    
    print("S :",S)
    print("D :",D)
    denominator = (n * (n-1))/2
    
    return (S + D)/denominator

#$#############################################################################    
# reading the dataset
data = pd.read_csv('online_shoppers_intention.csv')

# checking the shape of the data
#print(data.head())
#print(data.shape)
#print(data.info())
#print(data.describe())

#plt.rcParams['figure.figsize'] = (18, 7)
#
#plt.hist(data['ProductRelated_Duration'], color = 'lightgreen')
#plt.title('Distribution of diff Traffic',fontsize = 30)
#plt.xlabel('ProductRelated_Duration', fontsize = 15)
#plt.ylabel('Count', fontsize = 15)

#plt.plot(data['ProductRelated_Duration'])
#plt.show()

#data['Administrative'].hist(bins = 30)

#sns.boxplot(y='ProductRelated_Duration',data = data)
#data['Administrative_Duration'].hist(bins = 30)
#plt.scatter(data['ProductRelated_Duration'], data['Revenue'])
#plt.show()

#Convert the bool columns from TRUE > 1 and FALSE > 0 for Weekend and Revenue
data["Weekend"] *= 1
data["Revenue"] *= 1

#Convert the categorical column Month to number using mean encoding
mean_encode = data.groupby('Month')['Revenue'].mean()
data.loc[:, 'Month'] = data['Month'].map(mean_encode)

#Remove outliers for few columns by Top coding/capping
data.loc[data.ProductRelated_Duration  > 30000, 'ProductRelated_Duration'] = 30000

#Convert the categorical column VisitorType to number using one hot encoding
data = pd.get_dummies(data, prefix=['Vis'], columns=['VisitorType'])   
data = pd.get_dummies(data, prefix=['OS'], columns=['OperatingSystems'])  
data = pd.get_dummies(data, prefix=['browser'], columns=['Browser'])  
data = pd.get_dummies(data, prefix=['reg'], columns=['Region'])  
data = pd.get_dummies(data, prefix=['tt'], columns=['TrafficType'])  

y = data["Revenue"]
X = data.drop(columns="Revenue")

##### Feature importance ######################################################
    
#bestfeatures = SelectKBest(score_func=chi2, k=4)
#fit = bestfeatures.fit(X,y)
#dfscores = pd.DataFrame(fit.scores_)
#dfcolumns = pd.DataFrame(X.columns)
#featureScores = pd.concat([dfcolumns, dfscores], axis = 1)
#featureScores.columns = ['Specs','Score']
#print(featureScores)
#
#print(featureScores.nlargest(19,'Score'))

########## k means X ##########################################################
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
labels = kmeans.predict(X)
#np.savetxt("kmeans.txt", labels)

kmeans_DBI = davies_bouldin_score(X, labels)
print("KMeans DBI score :",kmeans_DBI)

kmeans_RI = rand_index_score(y, labels)#calculate_rand_index(y, labels)#rand_index_score(y, labels)
print("KMeans RI score :",kmeans_RI)

##### Complete-Linkage Agglomerative nesting ##################################
clustering = AgglomerativeClustering(n_clusters=4, linkage="complete").fit_predict(X)
#np.savetxt("AgglomerativeClustering.txt", clustering)

agnest_DBI = davies_bouldin_score(X, clustering)
print("AgglomerativeClustering DBI score :",agnest_DBI)

agnest_RI = rand_index_score(y, clustering)#calculate_rand_index(y, clustering)#rand_index_score(y, clustering)
print("AgglomerativeClustering RI score :",agnest_RI)
































