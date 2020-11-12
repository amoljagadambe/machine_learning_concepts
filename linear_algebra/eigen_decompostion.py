"""
                        Eigendecomposition
Eigenvalues and eigenvectors are easy to find with Python and NumPy. Remember,
an eigenvector of a square matrix  A  is a nozero vector  v  such that multiplication by  A
alters only the scale of  v
                                Av=λv
The scalar  λ  is known as the eigenvalue corresponding to this eigenvector.
 """

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# find the eigenvalues and eigenvectors for a simple square matrix
A = np.diag(np.arange(1, 4))
"""
[[1 0 0]
 [0 2 0]
 [0 0 3]]
"""

eigenvalues, eigenvectors = np.linalg.eig(A)
"""
Eigenvalue's: [1. 2. 3.]
Eigenvectors:[[1. 0. 0.]
              [0. 1. 0.]
              [0. 0. 1.]]
"""

# the eigenvalue w[i] corresponds to the eigenvector v[:, i]
print('Eigenvalue: {}'.format(eigenvalues[1]))
print('Eigenvector: {}'.format(eigenvectors[:, 1]))
"""
Eigenvalue: 2.0
Eigenvector: [0. 1. 0.]
"""

# verify eigendecomposition - should return original matrix
matrix = np.matmul(np.diag(eigenvalues), np.linalg.inv(eigenvectors))
output = np.matmul(eigenvectors, matrix).astype(np.int)
"""
[[1 0 0]
 [0 2 0]
 [0 0 3]]
"""

# plot the eigenvectors
origin = [0,0,0]

fig = plt.figure(figsize=(18,10))
fig.suptitle('Effects of Eigenvalues and Eigenvectors')
ax1 = fig.add_subplot(121, projection='3d')

ax1.quiver(origin, origin, origin, eigenvectors[0, :], eigenvectors[1, :], eigenvectors[2, :], color = 'k')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])
ax1.set_zlim([-3, 3])
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')
ax1.view_init(15, 30)
ax1.set_title("Before Multiplication")

# multiply original matrix by eigenvectors
new_eig = np.matmul(A, eigenvectors)
ax2 = plt.subplot(122, projection='3d')

# plot the new vectors
ax2.quiver(origin, origin, origin, new_eig[0, :], new_eig[1, :], new_eig[2, :], color = 'k')

# plot the eigenvalues for each vector (the amount the vector should be scaled by)
ax2.plot((eigenvalues[0]*eigenvectors[0]), (eigenvalues[1]*eigenvectors[1]), (eigenvalues[2]*eigenvectors[2]), 'rX')
ax2.set_title("After Multiplication")
ax2.set_xlim([-3, 3])
ax2.set_ylim([-3, 3])
ax2.set_zlim([-3, 3])
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')
ax2.set_zlabel('Z axis')
ax2.view_init(15, 30)

# check the png file for plot
plt.savefig('eigen_vectors.png')
plt.close(fig)
