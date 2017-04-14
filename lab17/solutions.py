"""Volume II: Compressed Sensing.
<Name>Matthew Gong
<Class>
<Date>
"""

import numpy as np

from cvxopt import solvers
from cvxopt import matrix
from matplotlib import pyplot as plt
from camera import Camera
from scipy import linalg as la
from visualize2 import visualizeEarth


# Problem 1
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Return only the solution x (not any slack variable), as a flat NumPy array.

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        x ((n, ) ndarray): The solution to the minimization problem.
    """

    m,n = np.shape(A)

    let_c_I = np.ones(n)
    let_c_o = np.zeros(n)
    let_c = np.hstack((let_c_I,let_c_o))

    #making G = |-I, I|
    #           |-I,-I|    
    ident = np.eye(n)
    neg_pos = np.hstack((-ident, ident))
    neg_neg = np.hstack((-ident,-ident))
    G = np.vstack((neg_pos,neg_neg))

    #A_not = [0,A]
    zeroes = np.zeros((m,n))
    A_not = np.hstack((zeroes,A))


    h = np.zeros(2*n)

    #convert to CVXOPT matrices
    c_matrix = matrix(let_c)
    G_matrix = matrix(G)
    h_matrix = matrix(h)
    A_matrix = matrix(A_not)
    b_matrix = matrix(b)

    solution = solvers.lp(c_matrix,G_matrix,h_matrix,A_matrix,b_matrix)
    solutions_array = np.array(solution['x'])[n:]

    return solutions_array



# Problem 2
def prob2(filename='ACME.png'):
    """Reconstruct the image in the indicated file using 100, 200, 250,
    and 275 measurements. Seed NumPy's random number generator with
    np.random.seed(1337) before each measurement to obtain consistent
    results.

    Resize and plot each reconstruction in a single figure with several
    subplots (use plt.imshow() instead of plt.plot()). Return a list
    containing the Euclidean distance between each reconstruction and the
    original image.
    """

    l = []
    acme_img = 1 - plt.imread('ACME.png')[:,:,0]
    measurements = [100, 200, 250, 275]

    for i in xrange(len(measurements)):
        np.random.seed(1337)

        A = np.random.randint(low = 0, high = 2, size =(measurements[i],32**2))
        b = A.dot(acme_img.flatten())

        recon = l1Min(A,b)
        recon_img = np.reshape(recon, (32,32))


        plt.subplot(2,2,i+1)
        plt.imshow(recon_img)

        l.append(la.norm(recon_img - acme_img))
    
    plt.show()

    return l


# Problem 3
def prob3(filename="StudentEarthData.npz"):
    """Reconstruct single-pixel camera color data in StudentEarthData.npz
    using 450, 650, and 850 measurements. Seed NumPy's random number generator
    with np.random.seed(1337) before each measurement to obtain consistent
    results.

    Return a list containing the Euclidean distance between each
    reconstruction and the color array.
    """

    data = np.load('StudentEarthData.npz')

    faces = data ['faces']
    vertices = data['vertices']
    C = data['C']
    V = data['V']

    myCamera = Camera(faces, vertices, C)

    measurements = [250, 450, 550]
    distance = []

    for i in measurements:
        np.random.seed(1337)
        
        myCamera.add_lots_pic(i)
        A, b = myCamera.returnData()
        A_hat = A.dot(V)

        c_1 = l1Min(A_hat, b[:,0])
        c1_hat = V.dot(c_1)

        c_2 = l1Min(A_hat, b[:,1])
        c2_hat = V.dot(c_2)

        c_3 = l1Min(A_hat, b[:,2])
        c3_hat = V.dot(c_3)

        c_hat = np.column_stack((c1_hat, c2_hat, c3_hat))

        distance.append(la.norm(c_hat - C))
        visualizeEarth(faces, vertices, c_hat.clip(0,1))

    visualizeEarth(faces, vertices, C)

    return distance

prob3()