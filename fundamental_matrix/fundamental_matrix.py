import cv2 
import numpy as np
from matplotlib import pyplot as plt

image1 = cv2.imread("/home/sailesh/sensor/fundamental_matrix/img1.jpeg")
image2 = cv2.imread("/home/sailesh/sensor/fundamental_matrix/img2.jpeg")
cv2.waitKey(0)

# Initiate ORB detector
oriented_BRIEF = cv2.ORB_create()
# find the keypoints with ORB
key_pts1 = oriented_BRIEF.detect(image1,None)
key_pts2 = oriented_BRIEF.detect(image2,None)

key_pts1, des1 = oriented_BRIEF.compute(image1, key_pts1)
key_pts2, des2 = oriented_BRIEF.compute(image2, key_pts2)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

bf_matcher = cv2.BFMatcher()
match = bf_matcher.knnMatch(des1,des2, k=2)

match_fit = []
pts1 = []
pts2 = []

for i,(m,n) in enumerate(match):
    if m.distance < 0.8*n.distance:
        match_fit.append(m)
        pts1.append(key_pts1[m.queryIdx].pt)
        pts2.append(key_pts2[m.trainIdx].pt)
        

points1 = np.int32(pts1)
points2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(points1,points2,cv2.FM_LMEDS)

print(F)



