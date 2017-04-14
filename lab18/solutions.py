"""Volume II Lab 18: Conjugate Gradient
<Name> Matthew Gong
<Class>
<Date>
"""

import numpy as np

from scipy import linalg as la
from scipy import optimize
from matplotlib import pyplot as plt



# Problem 1
def conjugateGradient(b, x_0, Q, tol=1e-4):
    """Use the Conjugate Gradient Method to find the solution to the linear
    system Qx = b.
    
    Parameters:
        b  ((n, ) ndarray)
        x0 ((n, ) ndarray): An initial guess for x.
        Q  ((n,n) ndarray): A positive-definite square matrix.
        tol (float)
    
    Returns:
        x ((n, ) ndarray): The solution to the linear systm Qx = b, according
            to the Conjugate Gradient Method.
    """

    r_0 = Q.dot(x_0) - b
    d_0 = -r_0

    while la.norm(r_0) > tol:
        a_0 = np.transpose(r_0).dot(r_0) / np.transpose(d_0).dot(Q).dot(d_0)

        x_1 = x_0 + a_0*d_0
        r_1 = r_0 + np.dot(a_0, Q).dot(d_0)
        b_1 = np.dot(r_1, r_1) / np.dot(r_0,r_0)
        d_1 = -r_1 + np.dot(b_1, d_0)
        
        d_0, r_0, x_0 = d_1, r_1, x_1

    print x_1
    return x_1




# Problem 2
def prob2(filename='linregression.txt'):
    """Use conjugateGradient() to solve the linear regression problem with
    the data from linregression.txt.
    Return the solution x*.
    """

    A = np.loadtxt(filename)
    b = A[:, 0].copy()

    A[:,0] = 1
    
    Q = np.dot(A.T,A)
    print Q
    b_prime = np.dot(A.T,b)
    x_0 = np.random.random(len(b_prime))

    
    return conjugateGradient(b_prime, x_0, Q)




# Problem 3
def prob3(filename='logregression.txt'):
    """Use scipy.optimize.fmin_cg() to find the maximum likelihood estimate
    for the data in logregression.txt.
    """

    def objective(b):
        return (np.log(1+np.exp(x.dot(b))) - y * x.dot(b)).sum()

    x = np.loadtxt(filename)

    m, n = x.shape

    y = x[:, 0].copy()
    x[:,0] = 1
    guess = np.random.random(n)

    b = optimize.fmin_cg(objective, guess)
    
    return b
