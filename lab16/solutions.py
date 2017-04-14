"""Volume II Lab 16: Simplex
<name> Matt Gong
<class>
<date>

Problems 1-6 give instructions on how to build the SimplexSolver class.
The grader will test your class by solving various linear optimization
problems and will only call the constructor and the solve() methods directly.
Write good docstrings for each of your class methods and comment your code.

prob7() will also be tested directly.
"""

import numpy as np



# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        maximize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    
    def __init__(self, c, A, b):
        """

        Parameters:
            c (1xn ndarray): The coefficients of the linear objective function.
            A (mxn ndarray): The constraint coefficients matrix.
            b (1xm ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """

        #checks validity of inputs
        if np.all(np.less_equal(np.zeros(len(b)), b)):
            self.c = c 
            self. A = A
            self.b = b
        else:
            raise ValueError('Problem isn\'t feasible at the origin')

        self.m = np.shape(self.A)[0]
        self.n = np.shape(self.A)[1]

        self.index_list = range(self.n, self.n + self.m) + range(self.n)

    #prob 3
    def make_table(self):
       
       #necessary steps to calculate the table intial table
       table = np.concatenate((self.A, np.eye(self.m)), axis = 1)
       top_row = np.concatenate((np.zeros(1), -self.c, np.zeros(self.m), np.array([1])))
       bot_row = np.concatenate((np.vstack(self.b), table, np.vstack(np.zeros(self.m))), axis = 1)
       
       #make table
       self.table = np.vstack((top_row, bot_row))



    def pivot_find(self):
        p = 0

        for i in self.table[0,1:]:
            if i < 0:
                break
            else:
                p += 1
                if p > self.m + self.n:
                    return -1, -1
        
        entering_var = p + 1

        b = self.table[1:,0]
        pivot_column = self.table[1:, entering_var]

        ratio = b/pivot_column

        if ratio.max() <= 0:
            raise ValueError('unbounded')

        for i in xrange(len(ratio)):
            if ratio[i] <= 0:
                ratio[i] = 1e10

        leaving_var = ratio.argmin() + 1

        return entering_var, leaving_var


    def pivot_swap(self):
        
        while self.pivot_find() != (-1,-1):

            entering_var , leaving_var = self.pivot_find()
            coeff_O = self.table[leaving_var][entering_var]
            
            self.table[leaving_var, : ] = self.table[leaving_var, : ] / coeff_O

            for i in range(self.m+1):

                if i != leaving_var:
                    coeff_m = self.table[leaving_var,entering_var] * -self.table[i, entering_var]
                    self.table[i,:] = coeff_m*self.table[leaving_var,:] + self.table[i,:]

        

    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The maximum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """

        self.make_table()
        self.pivot_find()
        self.pivot_swap()

        maximized = self.table[0,0]
        basic_vars = {}
        nonbasic_vars = {}

        objective = self.table[1:,0]
        for i in range(self.m + self.n):
            if i > self.m -1:
                nonbasic_vars[i] = 0
            constraint = self.table[:, i+1].argmax()
            basic_vars[i] = self.table[constraint,0]

        return maximized, basic_vars, nonbasic_vars


# Problem 7
def prob7(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        The minimizer of the problem (as an array).
    """

    data = np.load(filename)
    print data

    A = data['A']
    p = data['p']
    m = data['m']
    d = data['d']

    #add in positivity constraints
    A = np.vstack((A,np.eye(4)))
    b = np.hstack((m,d))

    D = SimplexSolver(p,A,b)

    D.make_table()

    D.pivot_swap()

    A, B, C = D.solve()

    x = []

    for i in xrange(len(D.c)):
        if i in B:
            x.append(B[i])
        elif i in C:
            x.append(0)


    return np.array(x)

# END OF FILE =================================================================

def test():
    c = np.array([3, 2])
    A = np.array([[1, -1], [3, 1], [4, 3]])
    b = np.array([2, 5, 7])
    D = SimplexSolver(c, A, b)
    #D.make_tableau()
    #print D.T
    #D.find_pivot()
    #D.swap_pivot()
    #print D.index_list
    D.solve()
test()
print prob7()