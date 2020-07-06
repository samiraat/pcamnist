

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import math
import operator
import cPickle
import gzip
import matplotlib.pyplot as plt
import seaborn as sns




def load_mnist_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return training_data

data = load_mnist_data()

X = data[0]
y = data[1]
data=np.zeros((1000,784))
for i in xrange(0,1000):
    data[i]=X[i]
#print data[0]



train = load_mnist_data()

mnist_X = train[0]
mnist_y = train[1]




def pca(k):
    m,n=data.shape
    mean=np.zeros((m,1))
    for i in range (0,m):
        mean[i]=np.mean(data[i,:])
    #print mean[0],mean[1],mean[2]

    #scatter matrix
    scatter_matrix=np.zeros((m,m))
    for i in range(0,n):
        scatter_matrix += (data[:,i].reshape(m,1) - mean).dot((data[:,i].reshape(m,1) - mean).T)
  #  print('Scatter Matrix:')
    #print scatter_matrix
    ####covarience matrix
    cov_matrix=np.zeros((m,m))
    for i in range(0,m):
        for j in range(0,m):
            np.cov([data[i,::],data[j,::]])
           # cov_matrix[i,j]=np.cov([data[i,::],data[j,::]])
   # print('Covariance Matrix:\n', cov_matrix)
    print np.cov([data[0,::],data[0,::]])
    # eigenvectors and eigenvalues for the from the scatter matrix
    eig_value_sc, eig_vector_sc = np.linalg.eig(scatter_matrix)
    print eig_value_sc
    # eigenvectors and eigenvalues for the from the covariance matrix
   # eig_value_cov, eig_vector_cov = np.linalg.eig(cov_matrix)

    for i in range(len(eig_value)):
        eigvec_sc = eig_vector_sc[:,i].reshape(1,m).T
        eigvec_cov = eig_vector_cov[:,i].reshape(1,m).T
        assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvectors are not identical'

    for ev in eig_vector_sc:
        numpy.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    # Make a list of (eigenvalue, eigenvector) tuples
    for i in range(len(eig_val_sc)):
        eig_pairs = (np.abs(eig_value_sc[i]), eig_vector_sc[:,i]) 

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: data[0], reverse=True)
    ###Choosing k eigenvectors with the largest eigenvalues
    matrix_w =np.zeros((k,1))
    for i in xrange(0,k):
        matrix_w[i] = eig_pairs[0][i].reshape(m,1)

    transformed = matrix_w.T.dot(data)
    print transformed
    
pca(40)
