import numpy as np
from matplotlib import pyplot as plt

# Load the 2D array from the .npy file
img_array = np.load('sino.npy')

# Display the array as a grayscale image
plt.imshow(img_array, cmap='gray')
plt.show()