"""
Normalization also called as norms have three types
    - L2 Norms   (also called as Euclidean Norm)
    - L1 Norms
    - Frobenius Norms (used for matrix normalization)

Normalization is nothing but calculating the magnitude of the given vector/matrix

"""
import numpy as np

# define a array
A = np.arange(9) - 3

# reshape array into 3x3 matrix
B = A.reshape((3, 3))

# Euclidean (L2) Norm - Default
print(np.linalg.norm(A))
print(np.linalg.norm(B))
"""
Output:
8.306623862918075
8.306623862918075
"""

# The Frobenius norm is the L2 norm for a Matrix
print(np.linalg.norm(B, 'fro'))
"""
Output:
8.306623862918075
"""

# the L1 norm
print(np.linalg.norm(A, 1))
print(np.linalg.norm(B, 1))
"""
Output:
21.0
8.0
"""

# the max norm (P = infinity)
print(np.linalg.norm(A, np.inf))
print(np.linalg.norm(B, np.inf))
"""
Output:
5.0
12.0
"""

"""
Vector Normalization
"""

# normalization to produce unit vector
norm = np.linalg.norm(A, 2)
A_unit = A / norm
"""
Output:
[-0.36115756 -0.24077171 -0.12038585  0.          0.12038585  0.24077171
  0.36115756  0.48154341  0.60192927]
"""

# the magnitude of a unit vector is equal to 1
print(np.linalg.norm(A_unit))
"""
Output:
1.0
"""