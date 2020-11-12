import numpy as np
import sys

print(f'Python Version is: {sys.version}')

# Define a Scalar
x = 6

# Define a Vector
z = np.array((1, 2, 3, 4))

print(f'Vector Dimension {z.shape}')
print(f'Vector size {z.size}')

# Define a matrix

m = np.array(([1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]))

print(f'Matrix Dimension {m.shape}')
print(f'Matrix size {m.size}')

one = np.zeros((3, 4), dtype=np.float)
"""
Output:

[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
"""

# Define a Tensor

tensor = np.ones((2, 2, 2), dtype=np.float)
"""
Output:
[[[1. 1.]
  [1. 1.]]

 [[1. 1.]
  [1. 1.]]]
"""

# Indexing

tensor[0, 1, 1] = 3.4
print(tensor)
"""
Output:
[[[1.  1. ]
  [1.  3.4]]

 [[1.  1. ]
  [1.  1. ]]]
"""

# Matrix Transpose

array = np.arange(6).reshape((2, 3))
"""
Output:
[[0 1 2]
 [3 4 5]]
"""
array = array.transpose()
"""
Output:
[[0 3]
 [1 4]
 [2 5]]
"""