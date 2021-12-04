import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
import pywt

start = time.time()

scale=0.25
dim_x=18000
dim_y=18000
grid_size= 3000
grid_amount=int((dim_x/grid_size)*(dim_y/grid_size))

scaled_x=int(dim_x*scale)
scaled_y=int(dim_y*scale)
scaled_grid_size=int(grid_size*scale)

img = cv.imread('scablands.jpg',1)
img = cv.resize(img, (scaled_x,scaled_y), interpolation=cv.INTER_AREA)


# Initiate SIFT detector
sift = cv.SIFT_create()
fast = cv.FastFeatureDetector_create()
orb = cv.ORB_create()

# find the keypoints and descriptors with SIFT
kp, des = sift.detectAndCompute(img,None)
img1 = cv.drawKeypoints(img, kp, None, color=(255,0,0))


plt.figure()
plt.imshow(img1)
plt.show()

end = time.time()
print(f"Runtime of the program is {end - start}")
