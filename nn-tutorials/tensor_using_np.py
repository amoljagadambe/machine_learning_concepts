import numpy as np

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

output_layer = np.dot(weights, inputs) + biases  # weights dim=(3,4) & inputs dim=(4,)
'''
Basic background for np.dot
np.dot(weights, inputs) = [np.dot(weights[0], inputs),np.dot(weights[1], inputs),np.dot(weights[2], inputs)]
'''
print(output_layer)  # Output_layer dim=(3,)
