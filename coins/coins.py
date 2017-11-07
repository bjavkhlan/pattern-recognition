import numpy as np
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import spectral_clustering

X = [[1, -1, 0, 0],[0, 0, 1, -1]]
def fill(i, j):
	global img
	global objs
	return

imgsrc = misc.imread('coins.jpg', mode='L')

th = 0
for i in range(len(imgsrc)):	
	for j in range(len(imgsrc[0])):
		th += imgsrc[i][j]
th /= len(imgsrc)*len(imgsrc[0])


!!!
bw = zeros(len(imgsrc), len(imgsrc[0]))



for i in range(len(imgsrc)):
	for j in range(len(imgsrc[0])):
		if imgsrc[i][j] < th:
			bw[i][j] = 0
		else: 
			bw[i][j] = 255 		
bw_med = ndimage.median_filter(bw, 3)

global img
global objs
img = bw_med



plt.imshow(imgsrc, cmap='gray')
plt.show()