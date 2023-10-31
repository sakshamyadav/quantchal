import qe_minesweeper
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import random 
import cv2
from scipy import ndimage as ndi
from skimage import data
from skimage.io import imshow, imread
from skimage.color import rgb2gray
from skimage import filters
from sklearn.cluster import DBSCAN
from ipywidgets import interact
from sklearn.preprocessing import StandardScaler
import scipy 
from findpeaks import findpeaks
from skimage.feature import peak_local_max
from skimage import data, color
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from sklearn.cluster import KMeans
from matplotlib.collections import PatchCollection
import skimage.io
import skimage.feature
import skimage.morphology
from matplotlib.collections import PatchCollection
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

scenario_num = random.randint(1,1000)
scenario_num = 788
magnetic_field, dipole = qe_minesweeper.load_dataset("C:\\Users\\saksh\\Desktop\\dataset\\stage1_test_dataset.h5", scenario_num)

mag_east = magnetic_field[0]
mag_north = magnetic_field[1]
mag_up = magnetic_field[2]


magnetic_field = magnetic_field.transpose(1,2,0)

magnetic_field = cv2.normalize(magnetic_field, None, 0, 255,cv2.NORM_MINMAX).astype(np.uint8)
image_gray = cv2.cvtColor(magnetic_field, cv2.COLOR_BGR2GRAY)
image_gray = cv2.bitwise_not(image_gray)
image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)

image_max = ndi.maximum_filter(image_gray, size=5, mode='constant')

blobs_log = blob_log(image_max, max_sigma=30, num_sigma=10, threshold=0.05)

# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
spots = blobs_log[:, :2]
spots = spots.transpose(1,0)
print(spots)
blobs_dog = blob_dog(image_gray, max_sigma=30)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    ax[idx].imshow(image_gray)
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax[idx].add_patch(c)
    ax[idx].set_axis_off()

plt.tight_layout()
plt.show()


#PLOT----------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(10, 4))

# Plot on the first subplot
plt.sca(axes[0])  # Set the current axes to the first subplot
im1 = plt.imshow(mag_up, cmap='hot', interpolation='nearest')
plt.colorbar(im1, ax=axes[0])  # Add a colorbar to the first subplot
axes[0].set_title('Magnetic Field (z)')

# Plot on the second subplot
plt.sca(axes[1])  # Set the current axes to the second subplot
im2 = plt.imshow(mag_east, cmap='hot', interpolation='nearest')
plt.colorbar(im2, ax=axes[1])  # Add a colorbar to the second subplot
axes[1].set_title('Magnetic Field (x)')

# Plot on the second subplot
plt.sca(axes[2])  # Set the current axes to the second subplot
im2 = plt.imshow(mag_north, cmap='hot', interpolation='nearest')
plt.colorbar(im2, ax=axes[2])  # Add a colorbar to the second subplot
axes[2].set_title('Magnetic Field (y)')

# Plot on the second subplot
plt.sca(axes[3])  # Set the current axes to the second subplot
im2 = plt.imshow(magnetic_field, cmap='hot', interpolation='nearest')
plt.colorbar(im2, ax=axes[3])  # Add a colorbar to the second subplot
axes[3].set_title('Magnetic Field as RGB (all channels)')

# Plot on the second subplot
plt.sca(axes[4])  # Set the current axes to the second subplot
plt.scatter(spots[1], spots[0], marker='x', color='red')
im2 = plt.imshow(magnetic_field)
plt.colorbar(im2, ax=axes[4])  # Add a colorbar to the second subplot
axes[4].set_title('gray')

# Add spacing between subplots
plt.tight_layout()

# Show the plots in the same window
plt.show()
