"""
posterior correlation estimation
"""
import numpy as np
from scipy.stats import spearmanr

a = np.loadtxt('posterior_regular_errfix.txt')
nparam = a.shape[1]

matrix = np.zeros((nparam-1,nparam-1))

for i in range(nparam):
    for j in range(i+1,nparam):
        (tmp,p) = spearmanr(a[:,i],a[:,j])
        if (p<1.0e-5):
            matrix[i,j-1] = tmp

print('covariance matrix: \n', matrix)
