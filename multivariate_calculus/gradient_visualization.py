import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# generate 2D meshgrid
nx, ny = (100, 100)

x = np.linspace(0, 10, nx)
y = np.linspace(0, 10, ny)

xv, yv = np.meshgrid(x, y)


# define function to plot
def f(x, y):
    return x * (y ** 2)


# calculate Z value for each X,Y point
z = f(xv, yv)

"""
Now that we have our mesh grid and have calculated f(x,y) for all points on the mesh 
grid,we can visualize the results! The z points on the graph will be represented using 
a colormap.
"""

# Make a Color plot to display the data
plt.figure(figsize=(14,12))
plt.pcolor(xv, yv, z)
plt.title('2D Color Plot of f(x,y)=xy^2')
plt.colorbar()
# plt.show()


# generate 2D meshgrid for Gradient
nx, ny = (10, 10)
x = np.linspace(0, 10, nx)
y = np.linspace(0, 10, ny)
xg, yg = np.meshgrid(x,y)

# calculate the gradient of f(x,y)
# Note: numpy returns answer in rows (y), columns (x) format
Gy, Gx = np.gradient(f(xg, yg))

# Make a Color plot to display the data
fig = plt.figure(figsize=(14,12))
plt.pcolor(xv, yv, z)
plt.colorbar()
plt.quiver(xg, yg, Gx, Gy, scale = 1000, color = 'w')
plt.title('Gradient of f(x,y) = xy^2')
plt.savefig('gradient.png')
plt.close(fig)
