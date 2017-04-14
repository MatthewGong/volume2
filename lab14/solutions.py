"""Volume 2 Lab 14: Optimization Packages II (CVXOPT)
<Name> Matthew Gong
<Class>
<Date>
"""
import numpy as np
from cvxopt import matrix
from cvxopt import solvers

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x + 2y          >= 3
                    2x + y + 3z     >= 10
                    x               >= 0
                    y               >= 0
                    z               >= 0

    Returns (in order):
        The optimizer (sol['x'])
        The optimal value (sol['primal objective'])
    """

    c = matrix([2.,1.,3.])
    G = matrix([[-1.,-2.,-1.,0.,0.],
        [-2.,-1.,0.,-1.,0.],
        [0.,-3.,0.,0.,-1.]])

    h = matrix([-3.,-10.,0.,0.,0.])

    sol = solvers.lp(c,G,h)

    return sol['x'], sol['primal objective']

def prob2():
    """Solve the transportation problem by converting all equality constraints
    into inequality constraints.

    Returns (in order):
        The optimizer (sol['x'])
        The optimal value (sol['primal objective'])

    minimize:       4x14 + 7x15 + 6x24 + 8x25 + 8x34 +9x35
    
    subject to:     x14 + x15       = 7
                    x24 + x25       = 2
                    x34 + x35       = 4
                    x14 + x24 + x34 = 5
                    x15 + x25 + x35 = 8

                    inequality version
                    x14 + x15                           < 7
                   -x14 - x15                           < -7
                                x24 + x25               < 2
                               -x24 - x25               < -2
                                            x34 + x35   < 4
                                           -x34 - x35   < -4
                    x14       + x24       + x34         < 5
                   -x14       - x24       - x34         < -5
                          x15       + x25       + x35   < 8
                         -x15       - x25       - x35   < -8
    """
    c = matrix([4., 7., 6., 8., 8., 9.])
    G = matrix([[1.,-1.,0.,0.,0.,0.,1.,-1.,0.,0.,-1.,0.,0.,0.,0.,0.],
                [1.,-1.,0.,0.,0.,0.,0.,0.,1.,-1.,0.,-1.,0.,0.,0.,0.],
                [0.,0.,1.,-1.,0.,0.,1.,-1.,0.,0.,0.,0.,-1.,0.,0.,0.],
                [0.,0.,1.,-1.,0.,0.,0.,0.,1.,-1.,0.,0.,0.,-1.,0.,0.],
                [0.,0.,0.,0.,1.,-1.,1.,-1.,0.,0.,0.,0.,0.,0.,-1.,0.],
                [0.,0.,0.,0.,1.,-1.,0.,0.,1.,-1.,0.,0.,0.,0.,0.,-1.],
        ])

    h = matrix([7.,-7.,2.,-2.,4.,-4.,5.,-5.,8,-8.,0.,0.,0.,0.,0.,0.,])

    """
    c = matrix([4., 7., 6., 8., 8., 9.])
    G = matrix(-1*np.eye(6))
    h = matrix(np.zeros(6))
    
    A = matrix([[1.,0.,0.,1.,0.],
                [1.,0.,0.,0.,1.],
                [0.,1.,0.,1.,0.],
                [0.,1.,0.,0.,1.],
                [0.,0.,1.,1.,0.],
                [0.,0.,1.,0.,1.]])

    b = matrix([7., 2., 4., 5., 8.])
    """
    print np.shape(c),np.shape(G),np.shape(h)
    sol = solvers.lp(c,G,h)

    return sol['x'], sol['primal objective']


def prob3():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer (sol['x'])
        The optimal value (sol['primal objective'])
    """
    
    Q = matrix([[3.,2.,1.],
                [2.,4.,2.],
                [1.,2.,3.]])

    p= matrix([3.,0.,1.])

    sol = solvers.qp(Q,p)

    return sol['x'],sol['primal objective']


def prob4():
    """Solve the allocation model problem in 'ForestData.npy'.
    Note that the first three rows of the data correspond to the first
    analysis area, the second group of three rows correspond to the second
    analysis area, and so on.

    21 variables in objective function

        
          7 equality location constraints 
  = 14 inequality constraints
    1 timber constraint
    1 grazing constraint
    1 wilderness constraint
    21 variable constraints < 0

    38 total constraints

    Returns (in order):
        The optimizer (sol['x'])
        The optimal value (sol['primal objective']*-1000)
    """

    data = np.load('ForestData.npy')

    
    objective = matrix(data[:,3])

    #print objective

    """
    create the contraint matrix
    """    

    #create a base constraint matrix for the (2*7)triplet sums paired inequality
    pi = np.zeros((14,21))

    #fill in the triplet sum matrix
    for i in range(0,21,3):
        j = i/3
        #positive inequality (7)
        pi[j,i] = 1.
        pi[j,i+1] = 1.
        pi[j,i+2] = 1.
        #negative inequality (7)
        pi[j+7,i] = -1.
        pi[j+7,i+1] = -1.
        pi[j+7,i+2] = -1.

    #collect the data from the data set for the (3) timber, grazing and wilderness constraints
    t = data[:,4]
    g = data[:,5]
    w = data[:,6]*(1./788.)

    #stack them for easier use later
    k = np.vstack((-t,-g,-w))

    #create the identity matrix for the -Xij<0 (21)
    padded = -1*np.eye(21)

    #put all of three part together to create the constraint matrix
    constraints = matrix(np.concatenate((pi,k,padded),axis=0))

    #print constraints

    """
    create the condition vector in the same order as the constraint matrix
    14 triplet summations
        7 positive
        7 negative
    3 given constraints
    21 Xij constraints
    """
    #the triplet summation we use the positive and negative version later
    a = data[::3,1]

    #the given constraints
    given = np.array([-40000.,-5.,-70.])

    #zeros created in function
    conditions = matrix(np.concatenate((a,-a, given, np.zeros(21))))
    
    print np.shape(objective),np.shape(constraints),np.shape(conditions)

    sol = solvers.lp(-objective,constraints,conditions)

    print sol['x'],sol['primal objective']*-1000.
    return sol['x'],sol['primal objective']*-1000.