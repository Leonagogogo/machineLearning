from skimage import io as imageio
import sys
import numpy as np
from sklearn.decomposition import PCA

# get the input original image and the K value (number of colors/clusters)
imgName = sys.argv[1]

# number of the pricinpal components
k = int(sys.argv[3])

# read in the image
image = imageio.imread(imgName)

# get the dimensions of the image
width, height, depth = image.shape

image_arr = np.reshape(image,(width,depth*height))

# apply PCA on each color component
pca = PCA(n_components=k).fit(image_arr)
compressed_arr = pca.transform(image_arr)
compressed_image = pca.inverse_transform(compressed_arr)

#reconstruct compressed image
compressed_image = compressed_image.reshape(width,height,depth).astype(np.uint8)

# save the compressed image
imageio.imsave(sys.argv[2],compressed_image)