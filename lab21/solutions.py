"""Volume II: Interior Point II (Quadratic Optimization).
<Name> Matt Gong
<Class>
<Date>
"""

from __future__ import division

import numpy as np

from scipy import linalg as la
from scipy.sparse import spdiags

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from cvxopt import matrix
from cvxopt import solvers



# Auxiliary function for problem 2
def startingPoint(G, c, A, b, guess):
    """
    Obtain an appropriate initial point for solving the QP
    .5 x^T Gx + x^T c s.t. Ax >= b.
    Inputs:
        G -- symmetric positive semidefinite matrix shape (n,n)
        c -- array of length n
        A -- constraint matrix shape (m,n)
        b -- array of length m
        guess -- a tuple of arrays (x, y, mu) of lengths n, m, and m, resp.
    Returns:
        a tuple of arrays (x0, y0, l0) of lengths n, m, and m, resp.
    """
    m,n = A.shape
    x0, y0, l0 = guess

    N = np.zeros((n+m+m, n+m+m))
    N[:n, :n]       = G
    N[:n, n+m:]     = -A.T
    N[n:n+m, :n]    = A
    N[n:n+m, n:n+m] = -np.eye(m)
    N[n+m:, n:n+m]  = np.diag(l0)
    N[n+m:, n+m:]   = np.diag(y0)


    rhs = np.empty(n+m+m)
    rhs[:n] = -(G.dot(x0) - A.T.dot(l0)+c)
    rhs[n:n+m] = -(A.dot(x0) - y0 - b)
    rhs[n+m:] = -(y0*l0)

    sol = la.solve(N, rhs)

    dx = sol[:n]
    dy = sol[n:n+m]
    dl = sol[n+m:]

    y0 = np.maximum(1, np.abs(y0 + dy))
    l0 = np.maximum(1, np.abs(l0+dl))

    return x0, y0, l0

# Problems 1-2
def qInteriorPoint(Q, c, A, b, guess, niter=20, tol=1e-16, verbose=False):
    """Solve the Quadratic program min .5 x^T Q x +  c^T x, Ax >= b
    using an Interior Point method.

    Parameters:
        Q ((n,n) ndarray): Positive semidefinite objective matrix.
        c ((n, ) ndarray): linear objective vector.
        A ((m,n) ndarray): Inequality constraint matrix.
        b ((m, ) ndarray): Inequality constraint vector.
        guess (3-tuple of arrays of lengths n, m, and m): Initial guesses for
            the solution x and lagrange multipliers y and eta, respectively.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    counter = 0
    nu = 100
    x,y,mu = startingPoint(Q,c,A,b, guess)

    m,n = A.shape

    e_vec = np.ones(m)

    while nu > tol and counter < niter:
        F = np.hstack((Q.dot(x) - A.T.dot(mu) + c, A.dot(x) - y - b, np.diag(y).dot(np.diag(mu).dot(e_vec))))


        #derivatives-ish
        Df1 = np.hstack((Q,np.zeros((n,m)), -A.T))
        Df2 = np.hstack((A, -np.eye(m), np.zeros((m,m))))
        Df3 = np.hstack((np.zeros((m,n)), np.diag(mu), np.diag(y)))
        df = np.vstack((Df1,Df2,Df3))

        nu = np.dot(y.T, mu)/m
        centering_parameter = np.hstack([np.zeros(n),np.zeros(m), .1*nu*e_vec])

        

        #find search direction
        search_dir = la.lu_solve(la.lu_factor(df), (-F + centering_parameter))    
        
        dx, dy, dmu = search_dir[:n], search_dir[n:n+m], search_dir[n+m:]

        mu_mask = search_dir[n+m:] < 0
        y_mask = search_dir[n:n+m] < 0

        if np.any(mu_mask):
            bmax = min(1, np.min(-mu[mu_mask]/search_dir[n+m:][mu_mask]))
        else:
            bmax = 1.

        if np.any(y_mask):
            dmax = min(1, np.min(-y[y_mask]/search_dir[n:n+m:][y_mask]))
        else:
            dmax = 1.

        beta = min(1., .95*bmax)
        delta = min(1., .95*dmax)
        alpha = min(beta, delta)

        #set for next iteration
        x = x+ alpha*dx
        y = y+ alpha*dy
        mu = mu + alpha*dmu

        counter += 1

    return x, .5*x.T.dot(Q.dot(x)) + c.T*x

# Auxiliary function for problem 3
def laplacian(n):
    """Construct the discrete Dirichlet energy matrix H for an n x n grid."""
    data = -1*np.ones((5, n**2))
    data[2,:] = 4
    data[1, n-1::n] = 0
    data[3, ::n] = 0
    diags = np.array([-n, -1, 0, 1, n])
    return spdiags(data, diags, n**2, n**2).toarray()

# Problem 3
def circus(n=15):
    """Solve the circus tent problem for grid size length 'n'.
    Plot and show the solution.
    """

    L = np.zerose((n,n))
    L[n//2-1:n//2+1, n//2-1:n//2+1] = .5

    m = [n//6-1, n//6, int(5*(n/6.))-1, int(5*(n/6.))]

    mask1, mask2 = np.meshgrid(m,m)

    l[mask1, mask2] = .3

    L_rav = L.ravel()
    x = np.ones((n,n)).ravel()
    y = np.ones(n**2)
    mu = np.ones(n**2)

    H = laplacian(n)
    A = np.eye(n**2)
    
    c = np.ones(n**2) - (n-1)**-2
    z = qInteriorPoint(H,c,A,L_rav, (x,y,mu))[0].reshape((n,n))


    domain = np.arange(n)
    X,Y = np.meshgrid(domain,domain)

    fig = plt.figure()
    ax1 = fig.add_subplot(111,projection = "3d")
    ax1.plot_surface(X,Y,z, rstride = 1, cstride = 1, color = 'r')
    plt.show()


# Problem 4
def portfolio(filename="portfolio.txt"):
    """Use the data in the specified file to estimate a covariance matrix and
    expected rates of return. Find the optimal portfolio that guarantees an
    expected return of R = 1.13, with and then without short selling.

    Returns:
        An array of the percentages per asset, allowing short selling.
        An array of the percentages per asset without allowing short selling.
    """


    with open(filename, 'r') as myfile:
        contents = myfile.readlines()

    #splits each line into it's parts
    for i in xrange(len(contents)):
        contents[i]  = contents[i].split()

        for j in xrange(len(contents[i])):
            contents[i][j] = float(contents[i][j])

    y = []
    x = []
    
    for k in range(len(contents)):
        y.append(contents[k][0])

    for h in range(len(contents)):
        x. append(contents[h][1:])


    x_arr = np.array(x)

    Q = matrix(np.cov(x_arr.T))
    mu = x_arr.mean(axis = 0)
    m,n = x_arr.shape

    A = matrix(np.array([np.ones(n), mu]))
    b = matrix(np.array([1.,1.3]))
    G = matrix(-np.eye(n))
    h = matrix(np.zeros(n))
    p = matrix(np.zeros(n))

    sol = solvers.qp(Q,p,G,h,A,b)

    return sol['x']
