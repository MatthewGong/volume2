"""Volume II Lab 19: Trust Region Methods
<Name> Matthew Gong
<Class>
<Date>
"""

import numpy as np

from scipy import linalg as la
from scipy import optimize as op



# Problem 1
def trustRegion(f,grad,hess,subprob,x0,r0,rmax=2.,eta=1./16,gtol=1e-5):
    """Implement the trust regions method.
    
    Parameters:
        f (function): The objective function to minimize.
        g (function): The gradient (or approximate gradient) of the objective
            function 'f'.
        hess (function): The hessian (or approximate hessian) of the objective
            function 'f'.
        subprob (function): Returns the step p_k.
        x0 (ndarray of shape (n,)): The initial point.
        r0 (float): The initial trust-region radius.
        rmax (float): The max value for trust-region radii.
        eta (float in [0,0.25)): Acceptance threshold.
        gtol (float): Convergence threshold.
        
    
    Returns:
        x (ndarray): the minimizer of f.
    
    Notes:
        The functions 'f', 'g', and 'hess' should all take a single parameter.
        The function 'subprob' takes as parameters a gradient vector, hessian
            matrix, and radius.
    """

    while la.norm(grad(x0)) > gtol:
         #print grad(x0), hess(x0), r0, 'start'
        p_k = subprob(grad(x0), hess(x0), r0)
        #print p_k,x0
        rho_k = (f(x0) - f(x0 + p_k)) / (-p_k.T.dot(grad(x0)) - .5*p_k.T.dot(hess(x0)).dot(p_k))
        # #print rho_k
        #print rho_k, f(x0+p_k) , 'b'
        if rho_k < 0.25:
            r1 = 0.25*r0
        else:
            if rho_k > 0.75 and la.norm(p_k) == r0:
                r1 = min(2*r0, rmax)
            else:
                r1 = r0


        if rho_k > eta:
            x1 = x0 + p_k
        else:
            x1 = x0


        #set for the next iteration
        x0 = x1
        #print x0
        r0 = r1
  
    return x0

# Problem 2   
def dogleg(gk,Hk,rk):
    """Calculate the dogleg minimizer of the quadratic model function.
    
    Parameters:
        gk (ndarray of shape (n,)): The current gradient of the objective
            function.
        Hk (ndarray of shape (n,n)): The current (or approximate) hessian.
        rk (float): The current trust region radius
    
    Returns:
        pk (ndarray of shape (n,)): The dogleg minimizer of the model function.
    """

    sol = np.linalg.solve(-Hk,gk)
    #print -Hk, gk,rk, 'vars'
    #print sol, 'sol'
    p_vec = -(np.transpose(gk).dot(gk)) / (np.dot(np.dot(np.transpose(gk), Hk), gk))

    U = p_vec*gk
     #print U, 'u'
    a_vec = np.transpose(sol).dot(sol) - 2*np.transpose(sol).dot(U) + np.transpose(U).dot(U)
    b_vec = 2* np.transpose(sol).dot(U) - 2 * np.transpose(U).dot(U)
    c_vec = np.transpose(U).dot(U) - rk**2

    #print 'norms', la.norm(sol), la.norm(U)
    if la.norm(sol) <= rk:
        return sol

    elif la.norm(U) >= rk:
        return rk*(U)/la.norm(U)

    else:
        x = np.max(np.roots([a_vec,b_vec,c_vec]))
         #print x
        T = x +1

        return U + (T-1)*(sol - U)




# Problem 3
def problem3():
    """Test your trustRegion() method on the Rosenbrock function.
    Define x0 = np.array([10.,10.]) and r = .25
    Return the minimizer.
    """
    
    x = np.array([10., 10.])
    rmax = 2.
    r = 0.25

    eta = 1./16.
    tol = 1e-5

    opts = {'initial_trust_radius':r, 'max_trust_radius': rmax, 'eta': eta, 'gtol': tol}

    sol1 = op.minimize(op.rosen, x, method = 'dogleg', jac = op.rosen_der, hess = op.rosen_hess, options = opts)
    sol2 = trustRegion(op.rosen, op.rosen_der, op.rosen_hess, dogleg, x, r, rmax, eta, gtol = tol)

    # #print np.allclose(sol1, sol2)

    return sol1, sol2


# Problem 4
def problem4():
    """Solve the described non-linear system of equations.
    Return the minimizer.
    """
    
    def l(x):
        return np.array([np.sin(x[0])*np.cos(x[1]) - 4*np.cos(x[0])*np.sin(x[1]), np.sin(x[1])*np.cos(x[0]) - 4*np.cos(x[1])*np.sin(x[0])])

    def f(x):
        return .5 * (l(x)**2).sum()

    def J(x):
        return np.array([[ np.cos(x[0])*np.cos(x[1]) + 4*np.sin(x[0])*np.sin(x[1]), -np.sin(x[0])*np.sin(x[1]) - 4*np.cos(x[0])*np.cos(x[1])],
                         [-np.sin(x[1])*np.sin(x[0]) - 4*np.cos(x[1])*np.cos(x[0]),  np.cos(x[1])*np.cos(x[0]) + 4*np.sin(x[1])*np.sin(x[0])]])

    def g(x):
        return J(x).dot(l(x))

    def H(x):
        return J(x).T.dot(J(x))


    rmax = 2.
    r = .25

    eta = 1./16
    tol = 1e-5

    x = np.array([3.5, -2.5])
    xs = trustRegion(f,g,H,dogleg,x,r,rmax,eta = eta, gtol = tol)

    return xs

problem3() 
print problem4()