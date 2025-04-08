# This code tests that the "demo.py" file properly solved the 
# 3-by-3 system of equations in the script.
#
# Author: Mike Rushka
# Date created: 4/08/2025
# Date modified: 4/08/2025

import numpy as np
from demo import A, b, x  # Import variables from your main script

# Compute Ax
Ax = A.dot(x)+1

# Check if Ax is approximately equal to b
assert np.allclose(Ax, b, atol=1e-8), f"❌ Test failed: Ax != b\nAx = {Ax}\nb = {b}"

print("✅ Test passed: Ax is approximately equal to b.")
