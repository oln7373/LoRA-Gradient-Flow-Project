# This code performs a simple matrix solve to demonstrate the use 
# of the Scipy package to perform matrix operations. The code is uploaded
# to Github and tested on the Github server.
#
# Author: Mike Rushka
# Date created: 4/08/2025
# Date modified: 4/08/2025


import numpy as np
from scipy import sparse
import scipy as sp
from scipy.sparse.linalg import splu  # LU solver for sparse matrices

# Ax = b

A = np.array([
    [3, 4, -2],
    [2, -3, 4],
    [1, -2, 3]
])

A = sp.sparse.csc_matrix(A)  # Convert A to sparse CSC format

b = np.array([0, 11, 7])  # Right-hand side as 1D array

# Perform LU decomposition and solve
lu = splu(A)
x = lu.solve(b)

print("Solution x:\n", x)