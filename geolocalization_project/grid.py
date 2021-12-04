import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
import pywt

start = time.time()

scale=0.5
dim_x=18000
dim_y=18000
grid_size= 3000
grid_amount=int((dim_x/grid_size)*(dim_y/grid_size))

scaled_x=int(dim_x*scale)
scaled_y=int(dim_y*scale)
scaled_grid_size=int(grid_size*scale)

img = cv.imread('sweden_full.png',1)
img = cv.resize(img, (scaled_x,scaled_y), interpolation=cv.INTER_AREA)


# Initiate SIFT detector
sift = cv.SIFT_create()
fast = cv.FastFeatureDetector_create()
orb = cv.ORB_create()

# find the keypoints and descriptors with SIFT
kp, des = sift.detectAndCompute(img,None)


grid_list=[]
for p in range(0, scaled_x -1, scaled_grid_size):

    cv.line(img, (p, 0), (p, scaled_y), (0, 0, 255), 2, int(scaled_y*0.1))
    cv.line(img, (0, p), (scaled_x, p), (0, 0, 255), 2, int(scaled_x*0.1))

for i in range(int(scaled_x/scaled_grid_size)):
    for j in range(int(scaled_y/scaled_grid_size)):
        grid_list.append([i*scaled_grid_size,(i+1)*scaled_grid_size,j*scaled_grid_size,(j+1)*scaled_grid_size])


print("Created {} grids each of size {}x{}".format(len(grid_list),scaled_grid_size,scaled_grid_size))
print("Classifying {} keypoints into {} grids".format(len(kp),len(grid_list)))
kp_save=[]

classification_count=np.zeros(grid_amount)
classification=[]
for i in kp:
    coordinate=[i.pt[0],i.pt[1]]
    kp_save.append(coordinate)
    for j in range(len(grid_list)):
        if coordinate[0]>=grid_list[j][0] and coordinate[0]<=grid_list[j][1] and coordinate[1]>=grid_list[j][2] and coordinate[1]<=grid_list[j][3]:
            classification.append(j)
            classification_count[j]=classification_count[j]+1

print("Finished classifying {} keypoints:".format(len(kp)))
for i in range(0,len(classification_count)):
    print("Grid {} - {} Keypoints".format(i,classification_count[i]))


np.savetxt('classification.csv', classification, delimiter=',')
print("Finished writing classification.csv")

np.savetxt('kp.csv', kp_save, delimiter=',')
print("Finished writing kp.csv")

np.savetxt('des.csv', des, delimiter=',')
print("Finished writing des.csv")



img1 = cv.drawKeypoints(img, kp, None, color=(255,0,0))


plt.figure()
plt.imshow(img1)
plt.show()

end = time.time()
print(f"Runtime of the program is {end - start}")
