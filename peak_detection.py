import qe_minesweeper
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import random 
import cv2
from scipy import ndimage as ndi
from skimage.io import imshow, imread
from skimage.color import rgb2gray
from skimage import filters
from sklearn.cluster import DBSCAN
from ipywidgets import interact
from sklearn.preprocessing import StandardScaler
import scipy 
from findpeaks import findpeaks
from skimage.feature import peak_local_max
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from sklearn.cluster import KMeans
from matplotlib.collections import PatchCollection
import skimage.io
import skimage.feature
import skimage.morphology
from matplotlib.collections import PatchCollection

scenario_num = random.randint(1,1000)
scenario_num = 423
magnetic_field, dipole = qe_minesweeper.load_dataset("C:\\Users\\saksh\\Desktop\\dataset\\stage1_training_dataset.h5", scenario_num)

mine_locations = qe_minesweeper.load_answers("C:\\Users\\saksh\\Desktop\\dataset\\stage1_training_dataset.h5", scenario_num)
mine_locations = mine_locations.astype(int)
mag_east = magnetic_field[0]
mag_north = magnetic_field[1]
mag_up = magnetic_field[2]

mine_loc_east = mine_locations[0]
mine_loc_north = mine_locations[1] 

for i in range(mine_locations.shape[1]):
    print('mine at', mine_loc_east[i], mine_loc_north[i])


magnetic_field = magnetic_field.transpose(1,2,0)
magnetic_field = cv2.normalize(magnetic_field, None, 0, 255,cv2.NORM_MINMAX).astype(np.uint8)
image_gray = cv2.cvtColor(magnetic_field, cv2.COLOR_BGR2GRAY)
print(image_gray.min(), image_gray.max())

#image_gray = cv2.normalize(image_gray, None, 0, 255,cv2.NORM_MINMAX).astype(np.uint8)
image_gray = cv2.bitwise_not(image_gray)
#image_gray = cv2.GaussianBlur(image_gray, (5,5), 0)
mlt = mine_locations.transpose(1,0)

def plotRoi(spots, img_ax, color, radius):
    patches1 = []
    patches2 = []
    for spot in spots:
        y, x = spot
        c1 = plt.Circle((x, y), radius)
        patches1.append(c1)
    img_ax.add_collection(PatchCollection(patches1, facecolors = "green", edgecolors = 'green', alpha = 0.3, linewidths = 5))

    
    for i in mlt:
        y, x = i 
        c2 = plt.Circle((x,y),radius)
        patches2.append(c2)
    img_ax.add_collection(PatchCollection(patches2, facecolors = "red", edgecolors = 'red', alpha = 0.3, linewidths = 1))



im = image_gray
image_max = ndi.maximum_filter(im, size=5, mode='constant')
fig, ax = plt.subplots()

ax.imshow(im, cmap = "Greys")
# Comparison between image_max and im to find the coordinates of local maxima
spots = peak_local_max(image_max, min_distance=9, exclude_border=True, threshold_abs=150)
plotRoi(spots, ax, "red", radius = 4)
#spots = spots.transpose(1,0)

print(spots)
# display results
fig, axes = plt.subplots(1, 4, figsize=(8, 3), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(im, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Original')

ax[1].imshow(image_max, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('Maximum filter')

ax[2].imshow(im, cmap=plt.cm.gray)
ax[2].autoscale(False)
ax[2].plot(spots[:, 1], spots[:, 0], 'r.')
ax[2].axis('off')
ax[2].set_title('Peak local max')

ax[3].imshow(im, cmap=plt.cm.gray)
ax[3].autoscale(False)
ax[3].plot(mlt[:, 1], mlt[:, 0], 'r.')
ax[3].axis('off')
ax[3].set_title('Actual')

fig.tight_layout()

plt.show()

spots = spots.transpose(1,0)
#PLOT----------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(10, 4))

# Plot on the first subplot
plt.sca(axes[0])  # Set the current axes to the first subplot
plt.scatter(mine_locations[1], mine_locations[0], marker='x', color='blue')
im1 = plt.imshow(mag_up, cmap='hot', interpolation='nearest')
plt.colorbar(im1, ax=axes[0])  # Add a colorbar to the first subplot
axes[0].set_title('Magnetic Field (z)')

# Plot on the second subplot
plt.sca(axes[1])  # Set the current axes to the second subplot
plt.scatter(mine_locations[1], mine_locations[0], marker='x', color='blue')
im2 = plt.imshow(mag_east, cmap='hot', interpolation='nearest')
plt.colorbar(im2, ax=axes[1])  # Add a colorbar to the second subplot
axes[1].set_title('Magnetic Field (x)')

# Plot on the second subplot
plt.sca(axes[2])  # Set the current axes to the second subplot
plt.scatter(mine_locations[1], mine_locations[0], marker='x', color='blue')
im2 = plt.imshow(mag_north, cmap='hot', interpolation='nearest')
plt.colorbar(im2, ax=axes[2])  # Add a colorbar to the second subplot
axes[2].set_title('Magnetic Field (y)')

# Plot on the second subplot
plt.sca(axes[3])  # Set the current axes to the second subplot
plt.scatter(mine_locations[1], mine_locations[0], marker='x', color='red')
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
