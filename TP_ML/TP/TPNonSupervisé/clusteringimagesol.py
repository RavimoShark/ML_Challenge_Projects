"""Example of how to use clustering to compress images."""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

from kmeans_sol import kmeans

img = ndimage.imread('china.jpg')
plt.imshow(img)
n_rows, n_cols, n_colors = img.shape

X = img.reshape(-1, n_colors).astype(np.float)
n_clusters = 64

labels, centers, _ = kmeans(X, n_clusters=n_clusters, n_iter=500)

X_quant = np.empty(X.shape)
for label in range(n_clusters):
    X_quant[labels == label, :] = centers[label]

img_quant = X_quant.reshape(img.shape).astype(np.uint8)

plt.figure()
plt.imshow(img_quant)
plt.show()
