'''
The actual implementation of the CG solver.
'''

import numpy as np

def default_M(x):
    return np.copy(x)
def default_dot(a, b):
    return a.dot(np.conj(b))

class CG:
    '''
    A simple Preconditioner Conjugate gradient solver.

    Parameters
    ----------
    A : callable
        A function representing the A matrix, callable as A(x).
    b : array
        Vector representing the rhs of the equation system.
    x0 : array, optional
        Initial guess.
    M : callable, optional
        A function representing the preconditioner matrix.
    dot :  callable, optional
        Function, callable as dot(a, b) that return the inner product used in the solver.
    fcg : bool, optional
        Use the flexible CG algorithm (using the Polak–Ribière formula for beta), see Notay 2000
        (doi:10.1137/S1064827599362314) and wikipedia.

    Notes
    -----
    Taken mostly from the implementation in pixell. Extended to the flexible CG algorithm,
    '''
    
    def __init__(self, A, b, x0=None, M=default_M, dot=default_dot, fcg=False):
        """Initialize a solver for the system Ax=b, with a starting guess of x0 (0
        if not provided). Vectors b and x0 must provide addition and multiplication,
        as well as the .copy() method, such as provided by numpy arrays. The
        preconditioner is given by M. A and M must be functors acting on vectors
        and returning vectors. The dot product may be manually specified using the
        dot argument. This is useful for MPI-parallelization, for example."""
        # Init parameters
        self.A = A
        self.b = b # not necessary to store this. Delete?
        self.M = M
        self.dot = dot
        if x0 is None:
            # used to be np.zeros_like, but writing it like this makes it
            # numpy/cupy-agnostic for a tiny cost
            self.x = b * 0
            self.r = b.copy()
        else:
            self.x = x0.copy()
            self.r = b-self.A(self.x)
        # Internal work variables
        n = b.size
        z = self.M(self.r)
        self.rz = self.dot(self.r, z)
        self.rz0 = float(self.rz)

        self.fcg = fcg
        if self.fcg:
            self.z = z.copy()

        self.p = z
        self.i = 0
        self.err = np.inf

    def step(self):
        '''
        Take a single step in the iteration. Results in .x, .i and .err being updated.
        To solve the system, call step() in a loop until you are satisfied with the accuracy. The result
        can then be read off from .x.
        '''
        
        if self.fcg:
            self._step_fcg()
        else:
            self._step_cg()

    def _step_cg(self):
        # Full vectors: p, Ap, x, r, z. Ap and z not in memory at the
        # same time. Total memory cost: 4 vectors + 1 temporary = 5 vectors
        Ap = self.A(self.p)
        alpha = self.rz/self.dot(self.p, Ap)
        self.x += alpha*self.p
        self.r -= alpha*Ap
        del Ap
        z = self.M(self.r)
        next_rz = self.dot(self.r, z)
        self.err = next_rz/self.rz0
        beta = next_rz/self.rz
        self.rz = next_rz
        self.p = z + beta*self.p
        self.i += 1

    def _step_fcg(self):
        Ap = self.A(self.p)
        alpha = self.rz/self.dot(self.p, Ap)
        self.x += alpha*self.p
        self.r -= alpha*Ap
        del Ap
        next_z = self.M(self.r)
        next_rz = self.dot(self.r, next_z)
        self.err = next_rz/self.rz0        
        beta = self.dot(self.r, (next_z - self.z)) / self.rz        
        self.rz = next_rz
        self.z = next_z        
        self.p  = self.z + beta*self.p
        self.i += 1
        
