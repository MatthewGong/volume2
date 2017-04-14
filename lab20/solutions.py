"""Volume 2 Lab 19: Interior Point 1 (Linear Programs)
<Name> Matthew Gong
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from scipy.stats import linregress

from matplotlib import pyplot as plt


# Auxiliary Functions ---------------------------------------------------------
def startingPoint(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A.dot(A.T))
    x = A.T.dot(B.dot(b))
    lam = B.dot(A.dot(c))
    mu = c - A.T.dot(lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu

# Use this linear program generator to test your interior point method.
def randomLP(m):
    """Generate a 'square' linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add slack variables.
    Inputs:
        m -- positive integer: the number of desired constraints
             and the dimension of space in which to optimize.
    Outputs:
        A -- array of shape (m,n).
        b -- array of shape (m,).
        c -- array of shape (n,).
        x -- the solution to the LP.
    """
    n = m
    A = np.random.random((m,n))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(n)*10
    b = A.dot(x)
    c = A.sum(axis=0)/float(n)
    return A, b, -c, x

# This random linear program generator is more general than the first.
def randomLP2(m,n):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Inputs:
        m -- positive integer >= n, number of desired constraints
        n -- dimension of space in which to optimize
    Outputs:
        A -- array of shape (m,n+m)
        b -- array of shape (m,)
        c -- array of shape (n+m,), with m trailing 0s
        v -- the solution to the LP
    """
    A = np.random.random((m,n))*20 - 10
    A[A[:,-1]<0] *= -1
    v = np.random.random(n)*10
    k = n
    b = np.zeros(m)
    b[:k] = A[:k,:].dot(v)
    b[k:] = A[k:,:].dot(v) + np.random.random(m-k)*10
    c = np.zeros(n+m)
    c[:n] = A[:k,:].sum(axis=0)/k
    A = np.hstack((A, np.eye(m)))
    return A, b, -c, v


# Problems --------------------------------------------------------------------
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
    using an Interior Point method.

    Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """

    #necessary vars
    m,n = A.shape
    Df_a = np.bmat([[np.zeros((n,n)), A.T, np.eye(n)], [A, np.zeros((m,m)), np.zeros((m,n))]])


    def F(x, lam, mu):
        return np.hstack((np.dot(A.T, lam) + mu - c, np.dot(A,x)- b, np.diag(mu).dot(x))) 

    def Search_direction(x, mu, sigma = .1):
        Df_2 = np.bmat([[np.diag(mu), np.zeros((n,m)), np.diag(x)]])
        Df = np.vstack((Df_a,Df_2))
        mu_0 = x.dot(mu)/n

        centering_parameter = np.hstack((np.zeros(n), np.zeros(m), sigma*mu_0*np.ones(n)))

        DDf, piv = la.lu_factor(Df)

        return la.lu_solve((DDf,piv), -F(x, lam, mu) + centering_parameter)


    x, lam, mu = startingPoint(A, b, c)
    #print x, 'x'
    counter = 0

    while x.dot(mu)/n > tol and counter < niter:
        search_dir = Search_direction(x, mu)
        dx, dlam, dmu = search_dir[:n], search_dir[n:n+m], search_dir[n+m:]

        mu_mask = dmu < 0

        x_mask  = dx < 0

        if np.all(mu_mask) == False:
            amax = 1
        else:
            amax = np.min(np.hstack((1., -mu[mu_mask]/dmu[mu_mask])))

        if np.all(x_mask) == False:
            dmax = 1
        else:
            dmax = np.min(np.hstack((1., -x[x_mask]/dx[x_mask])))

        a = np.min([1., .95*amax])
        d = np.min([1., .95*dmax])

        x = x + d*dx
        lam = lam + a*dlam
        mu = mu + a*dmu

        counter += 1

    return x, np.dot(c,x)    


def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""
    
    data = np.loadtxt(filename)

    slope, intercept = linregress(data[:, 1], data[:,0])[:2]
    domain = np.linspace(0, 10, 200)

    plt.plot(domain, domain*slope + intercept)
    plt.scatter(data[:,1], data[:,0])

    #initial
    m,n = data.shape
    n -= 1

    c = np.zeros(3*m + 2*(n+1))
    c[:m] = 1

    y = np.empty(2*m)
    y[::2]  = -data[:, 0]
    y[1::2] =  data[:, 0]
    x       =  data[:, 1:]

    #constraints
    A = np.ones((2*m,3*m + 2*(n+1)))

    A[::2 , :m]      = np.eye(m)
    A[1::2, :m]      = np.eye(m)
    
    A[::2 , m:m+n]      = -x
    A[1::2, m:m+n]      =  x
    A[::2 , m+n:m+2*n]    =  x
    A[1::2, m+n:m+2*n]    = -x
 
    A[::2 , m+2*n]      = -1
    A[1::2, m+2*n+1]    = -1

    A[:   , m+2*n+2:]   = -np.eye(2*m,2*m)


    #calculate the solutions
    sol = interiorPoint(A,y,c,niter=10)[0]

    beta = sol[m:m+n] - sol[m+n:m+2*n]
    b = sol[m+2*n] - sol[m+2*n+1]
    yis = beta*x+b

    plt.plot(x,yis)
    plt.show()
