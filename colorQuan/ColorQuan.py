import numpy as np
from skimage import io
import urllib
import cv2
import sys


def covert_image(url, K):
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	img = cv2.imdecode(image, cv2.IMREAD_COLOR)
	Z = img.reshape((-1,3))
	# convert to np.float32
	Z = np.float32(Z)
	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))
	cv2.imshow('res2',res2)
	#io.imsave(path, res2)
	io.imsave(path,res2)

#read path from command line
k=sys.argv[3]
path = sys.argv[2]
url = sys.argv[1]
covert_image(url, int(k))




