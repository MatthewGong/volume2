# name this file 'solutions.py'.
"""Volume II Lab 15: Line Search Algorithms
<name> Matt Gong
<class>
<date> 
"""

import numpy as np
from scipy import linalg as la
from scipy.optimize import leastsq,line_search

from matplotlib import pyplot as plt



# Problem 1
def newton1d(f, df, ddf, x, niter=10):
    """
    Perform Newton's method to minimize a function from R to R.

    Parameters:
        f (function): The twice-differentiable objective function.
        df (function): The first derivative of 'f'.
        ddf (function): The second derivative of 'f'.
        x (float): The initial guess.
        niter (int): The number of iterations. Defaults to 10.
    
    Returns:
        (float) The approximated minimizer.
    """

    x_0 = x
    x_k = x

    for i in xrange(niter):
        x_k1 = x_k - df(x_k)/ddf(x_k)
        x_k = x_k1

    return x_k


def test_newton():
    """Use the newton1d() function to minimixe f(x) = x^2 + sin(5x) with an
    initial guess of x_0 = 0. Also try other guesses farther away from the
    true minimizer, and note when the method fails to obtain the correct
    answer.

    Returns:
        (float) The true minimizer with an initial guess x_0 = 0.
        (float) The result of newton1d() with a bad initial guess.
    """

    f = lambda x: x**2 + np.sin(5*x)
    df = lambda x: 2*x + 5*np.cos(5*x)
    ddf = lambda x: 2 + 0,-25*np.sin(5*x)


    print newtonsMethod(f,df,ddf, 0, niter = 100)

# Problem 2
def backtracking(f, slope, x, p, a=1, rho=.9, c=10e-4):
    """Perform a backtracking line search to satisfy the Armijo Conditions.

    Parameters:
        f (function): the twice-differentiable objective function.
        slope (float): The value of grad(f)^T p.
        x (ndarray of shape (n,)): The current iterate.
        p (ndarray of shape (n,)): The current search direction.
        a (float): The intial step length. (set to 1 in Newton and
            quasi-Newton methods)
        rho (float): A number in (0,1).
        c (float): A number in (0,1).
    
    Returns:
        (float) The computed step size satisfying the Armijo condition.
    """


    while f(x+a*p) > f(x) + c*a*slope :
        a = rho*a

    return a


# Problem 3    
def gradientDescent(f, df, x, niter=10):
    """Minimize a function using gradient descent.

    Parameters:
        f (function): The twice-differentiable objective function.
        df (function): The gradient of the function.
        x (ndarray of shape (n,)): The initial point.
        niter (int): The number of iterations to run.
    
    Returns:
        (list of ndarrays) The sequence of points generated.
    """

    points = []

    for i in xrange(niter):
        point = -dfx
        slope = np.dot(point,-point)
        
        #calculate a
        a = backtracking(f,slope,x,point)
        

        #update the search point
        x_k = x + a*p
        points.append(x_k)
        x = x_k

    return points

def newtonsMethod(f, df, ddf, x, niter=10):
    """Minimize a function using Newton's method.

    Parameters:
        f (function): The twice-differentiable objective function.
        df (function): The gradient of the function.
        ddf (function): The Hessian of the function.
        x (ndarray of shape (n,)): The initial point.
        niter (int): The number of iterations.
    
    Returns:
        (list of ndarrays) The sequence of points generated.
    """

    points = []

    for i in xrange(niter):
        point = np.dot(-la.inv(ddf(x)), (df(x)))

        slope = np.dot(df(x), point)

        a = backtracking(f, slope, x, point)
        
        #update point
        x_k = x + a*point
        points.append(x_k)
        x = x_k

    return points


# Problem 4
def gaussNewton(f, df, jac, r, x, niter=10):
    """Solve a nonlinear least squares problem with Gauss-Newton method.

    Parameters:
        f (function): The objective function.
        df (function): The gradient of f.
        jac (function): The jacobian of the residual vector.
        r (function): The residual vector.
        x (ndarray of shape (n,)): The initial point.
        niter (int): The number of iterations.
    
    Returns:
        (ndarray of shape (n,)) The minimizer.
    """

    for i in xrange(niter):
        #check if it's close enough
        if np.allclose(np.dot(jac(x).T, r(x)), 0):
            return x

        else:
            p = la.solve(np.dot(jac(x).T, jac(x)), -np.dot(jac(x).T, r(x)))

            a = line_search(f, df, x, p)[0]
            if a is None:
                return x
            else:
                x_k = x + a*p

    return x_k

# Problem 5
def census():
    """Generate two plots: one that considers the first 8 decades of the US
    Census data (with the exponential model), and one that considers all 16
    decades of data (with the logistic model).
    """

    # Start with the first 8 decades of data.
    years1 = np.arange(8)
    pop1 = np.array([3.929,  5.308,  7.240,  9.638,
                    12.866, 17.069, 23.192, 31.443])


    # Now consider the first 16 decades.
    years2 = np.arange(16)
    pop2 = np.array([3.929,   5.308,   7.240,   9.638,
                    12.866,  17.069,  23.192,  31.443,
                    38.558,  50.156,  62.948,  75.996,
                    91.972, 105.711, 122.775, 131.669])

    def model_1(x, t):
        return x[0]*np.exp(x[1]*(years1 + x[2]))

    def model_2(x,t):
        return x[0]/(1+np.exp(-x[1]*(years2 + x[2])))

    def residual_1(x):
            return model_1(x,years1) - pop1

    def residual_2(x):
            return model_2(x,years2) - pop2

    x0 = np.array([150, 0.4, 2.5])

    lsqrs_1 = leastsq(residual_1, x0)[0]

    plt.plot(years1, pop1, '*', label = 'data')
    plt.plot(years1, model_1(lsqrs_1,years1), label = 'moedl')

    plt.show()

    x0 = np.array([150, 0.4, -15])

    lsqrs_2 = leastsq(residual_2, x0)[0] 

    print lsqrs_2
    plt.plot(years2, pop2, '*', label = 'data')
    plt.plot(years2, model_2(lsqrs_2,years2), label = 'model')

    plt.show()

