# This code contains written function definitions for 
# various functions used in the "LoRA_gradient_descent.py"
# file.
#
# Author: Mike Rushka
# Date created: 4/09/2025
# Date modified: 4/09/2025


import numpy as np


# Computes the gradient of ||W + AB||_F^2 with respect to A
def grad_A(A, B, W):
    return 2 * (W + A @ B) @ B.T

# Computes the gradient of ||W + AB||_F^2 with respect to B
def grad_B(A, B, W):
    return 2 * A.T @ (W + A @ B)