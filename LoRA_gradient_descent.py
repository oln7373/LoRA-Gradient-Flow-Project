# This code performs the LoRA gradient descent algorithm on the toy problem
# of generating low-rank finetuning weights to minimize the Frobenius norm of 
# a randomly generated n by m matrix. The efficiency of the algorithm is compared 
# for varying values of k (see Algorithm 1 of Mike Rushka's research notes).
#
# Author: Mike Rushka
# Date created: 4/09/2025
# Date modified: 4/09/2025


import numpy as np
import scipy as sp
import time
from LoRA_gd_functions import grad_A, grad_B #import partial gradient functions from another file

#define constants
n = 8 #rows of W
m = 8 #columns of W
r = 1 #rank of AB
alpha = 1e-6 #learning rate
k = 100 #see algorithm 1 notation from notes
tol = 1e-8
high = 100 #range for uniform random integer generator 

np.random.seed(42) #seed np.random for debugging purposes

#randomly generate initial n x n matrix W, with each entry pulled from uniform distribution, W is a constant
W = np.random.randint(0, high = high, size = (n, m))

#randomly generate finetuning matrix, A and B will change
A = np.random.randint(0, high = high, size = ( n , r ))
B = np.random.randint(0, high = high, size = ( r , m ))

#main loop, this is the LoRA gradient descent algorithm as defined in section 2 of Mike's notes

iter = 0
norm_prev = np.linalg.norm(W + A @ B, 'fro')
diff = tol + 1 #force main loop to run at least once
start_time = time.time()  #start the clock
while (diff > tol): #convergence check included in the while loop statement

    for i in range(k):
        A = A - alpha*grad_A(A,B,W)

    for j in range(k):
        B = B - alpha*grad_B(A,B,W)

    iter += 1

    diff = abs(np.linalg.norm(W + A @ B, 'fro') - norm_prev)
    norm_prev = np.linalg.norm(W + A @ B, 'fro')

end_time = time.time()  #stop the clock

#print results
print("Iterations to converge:", iter)
print("Elapsed time (seconds):", end_time - start_time)
print("Total partial derivatives calculated:", r*k*iter*(m+n))
print("Final Frobenius norm:", np.linalg.norm(W + A @ B, 'fro'))
