import math
import numpy as np
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import spectral_clustering
from sklearn.cluster import KMeans
from operator import itemgetter

X = [[1, 0, 0],[0, 1, -1]]
quantity = [0, 0, 0, 0, 0]
quality = [1, 10, 50, 100, 500]

def fill(i, j, k):
	global img
	global objs
	global minx, maxx
	if objs[i][j] != 0:
		return
	objs[i][j] = k
	minx[k] = min(minx[k], i)
	maxx[k] = max(maxx[k], i)
	for r in range(len(X[0])):
		if i+X[0][r]<len(img) and i+X[0][r]>=0 and j+X[1][r]<len(img[0]) and j+X[1][r]>=0 and img[i+X[0][r]][j+X[1][r]]==255 and objs[i+X[0][r]][j+X[1][r]]==0:
			fill(i+X[0][r], j+X[1][r], k)
	return




imgsrc = misc.imread('coins.jpg', mode='L')

th = 0
for i in range(len(imgsrc)):	
	for j in range(len(imgsrc[0])):
		th += imgsrc[i][j]
                th /= len(imgsrc)*len(imgsrc[0])


bw = np.zeros( (len(imgsrc), len(imgsrc[0])) )


for i in range(len(imgsrc)):
	for j in range(len(imgsrc[0])):
		if imgsrc[i][j] < th:
			bw[i][j] = 0
		else: 
			bw[i][j] = 255 		
                        bw_med = ndimage.median_filter(bw, 9)

global img
global objs
global minx, maxx


cnt = 1
img = bw_med
objs = np.zeros( (len(img), len(img[0])), dtype=np.int )
minx = np.zeros( 50, dtype=np.int )
maxx = np.zeros( 50, dtype=np.int )

for i in range(len(minx)):
	minx[i] = len(img)

for i in range(len(img)):
	for j in range(len(img[0])):
		if img[i][j] == 255 and objs[i][j] == 0:
			fill(i, j, cnt)
			cnt += 1
			#print(i, j, cnt)
                        #hist, bin_edges = np.histogram(objs)

#print(cnt)

areas = np.zeros( cnt, dtype=np.int )
for i in range(len(objs)):
	 for j in range(len(objs[0])):
		 areas[int(objs[i][j])] += 1

rads = np.zeros( cnt, dtype=np.int )
for i in range(len(rads)):
	rads[i] = maxx[i]-minx[i]+1

#print(areas)
#print(rads)
notCircle = 0
eps = 0.15
coins = []
for i in range(len(areas)):
	relation = areas[i]/(math.pi*rads[i]*rads[i]*0.25)
	if relation > 1+eps or relation < 1-eps:
		#print(relation, areas[i])
		notCircle += 1
	else:
		coins.insert(len(coins), [areas[i], 0])


coins = np.array(coins)
#print(coins)

kmeans = KMeans(n_clusters=5, random_state=0).fit(coins)
#print(kmeans.labels_)
#print(kmeans.cluster_centers_[0][0])
for i in range(len(kmeans.labels_)):
	quantity[kmeans.labels_[i]] += 1
	#print(kmeans.labels_[i], coins[i][0])

coins = []
for i in range(len(kmeans.cluster_centers_)):
        #print(kmeans.cluster_centers_[i][0])
        coins.insert(0, [kmeans.cluster_centers_[i][0], quantity[i]])

coins.sort()
for i in range(len(coins)):
        quantity[i] = coins[i][1]
        
sum = 0
for i in range(len(quantity)):
	sum += quantity[i]*quality[i]
	print(quality[i], "-iin zoos ", quantity[i])

print("Total sum:", sum)

#print(notCircle)
plt.imshow(img, cmap='gray')
plt.show()
