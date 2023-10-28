import qe_minesweeper
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random 
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
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


result_array = []

for scenario_num in range(0,1000):
    magnetic_field, dipole = qe_minesweeper.load_dataset("C:\\Users\\saksh\\Desktop\\dataset\\stage1_test_dataset.h5", scenario_num)

    magnetic_field = magnetic_field.transpose(1,2,0)

    magnetic_field = cv2.normalize(magnetic_field, None, 0, 255,cv2.NORM_MINMAX).astype(np.uint8)
    image_gray = cv2.cvtColor(magnetic_field, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.bitwise_not(image_gray)
    image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)

    image_max = ndi.maximum_filter(image_gray, size=5, mode='constant')

    blobs_log = blob_log(image_max, max_sigma=30, num_sigma=10, threshold=0.05)
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
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot on the second subplot
    plt.sca(axes[0])  # Set the current axes to the second subplot
    im2 = plt.imshow(magnetic_field, cmap='hot', interpolation='nearest')
    plt.colorbar(im2, ax=axes[0])  # Add a colorbar to the second subplot
    axes[0].set_title('Magnetic Field as RGB (all channels)')

    # Plot on the second subplot
    plt.sca(axes[1])  # Set the current axes to the second subplot
    plt.scatter(spots[1], spots[0], marker='x', color='red')
    im2 = plt.imshow(magnetic_field)
    plt.colorbar(im2, ax=axes[1])  # Add a colorbar to the second subplot
    axes[1].set_title('gray')

    # Add spacing between subplots
    plt.tight_layout()

    # Show the plots in the same window
    plt.show()

    #result_array.append(spots.tolist())


#print(result_array[0])

#res = qe_minesweeper.submit_answers(result_array, '1', 'XRvCHL')
#print(res)