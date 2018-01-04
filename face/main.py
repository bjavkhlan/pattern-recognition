import glob
import math
import numpy as np
from numpy import linalg
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
from skimage import io
from skimage import exposure
import random

filepath = './Lab4/pie 10/pie 10/*.bmp'

K = 3
NofLabel = 15

data = []
label = []
for filename in glob.glob(filepath):
    image = misc.imread(filename, mode='L')
    '''
    Resize 
    '''
    image = misc.imresize(image, [50, 50])
    '''
    Histogram Equalization
    '''
    image = exposure.equalize_hist(image)
    lbl = filename[len(filename)-12]+filename[len(filename)-11]
    lbl = int(lbl)-55
    vector = []
    for row in image:
        for dot in row:
           vector.append(1.0*dot/255)
    data.append(vector)
    label.append(lbl)


'''
PCA 
'''
    
covar = np.cov(np.transpose(data))
[eigval, eigvec] = linalg.eig(covar)


idx = eigval.argsort()[::-1]   
eigval = eigval[idx]
eigvec = eigvec[:,idx]

neweig = np.zeros([len(eigvec), 50], dtype=np.complex128)
for i in range(len(neweig)):
    for j in range(len(neweig[i])):
        neweig[i][j] = eigvec[i][j]
        
newData = np.dot(data, neweig)


'''
Test Train
'''

test = []
train = []
testLabel = []
trainLabel = []

for i in range(150):
    if random.random() < 0.3:
        test.append(newData[i])
        testLabel.append(label[i])
    else:
        train.append(newData[i])
        trainLabel.append(label[i])

print('Number of test image:', len(test))




'''
KNN
'''
success = 0
accuracy = 0
for i in range(len(test)):
    dist = []
    for j in range(len(train)):
        dist.append([ linalg.norm(test[i]-train[j]), trainLabel[j] ])
    dist.sort()

    d = np.zeros([NofLabel+1])
    for j in range(K):
        d[dist[j][1]] += 1
    mx = 0
    for j in range(len(d)):
        if d[j] > d[mx]:
            mx = j

    accuracy += d[mx]
    if mx == testLabel[i]:
        success += 1


print('Accuracy from K: ', 1.0*accuracy/(len(test)*K))
print('Accuracy: ', 1.0*success/len(test))

