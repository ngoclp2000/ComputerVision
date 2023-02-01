from IPython.display import Image
from skimage.feature import hog

import skimage
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import imutils

template = cv2.imread('./templates/Soccer_ball.svg.png')
image = cv2.imread('./images/28-4-2021.jpg')

templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(16,8))


# plt.subplot(221), plt.imshow(templateGray)
# plt.subplot(222), plt.imshow(imageGray)
# plt.show()

# tạo bộ trích chọn đặc trưng
sift = cv2.SIFT_create()

# tìm điểm đặc trưng và tính sift cho từng ảnh
kp1, des1 = sift.detectAndCompute(template,None)
kp2, des2 = sift.detectAndCompute(image,None)

## tìm các cặp đặc trưng tương đồng giữa 2 ảnh. Có thể sử dụng phương pháp vét cạn BRUTE_FORCE matching hoặc FLANN hỗ trợ so khớp nhanh hơn
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches2 = flann.knnMatch(des1,des2,k=2) # trả về 2 features gần nhất từ tập des2

MIN_MATCH_COUNT=10
good = []
distance_ratio = 0.7
for m,n in matches2:
    if m.distance < distance_ratio*n.distance: # chỉ giữ lại những ghép cặp ổn định (m.distance: khoảng cách gần nhất, n.distance: khoảng cách gần thứ 2)
        good.append(m)

## khoanh vùng đối tượng tương ứng được tìm thấy
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
    # tìm phép biến đổi đồng dạng giữa 2 tập điểm được so khớp
    ### YOUR CODE HERE
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) 
    ### YOUR CODE HERE
    
    matchesMask = mask.ravel().tolist()
    h,w,d = template.shape
    
    # xác định vị trí tương ứng trên ảnh đích của 4 góc trên ảnh nguồn
    
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2) # 4 điểm khoanh vùng trên ảnh nguồn
    ### YOUR CODE HERE
    dst = cv2.perspectiveTransform(pts,M) # thực hiện biến đổi để tìm ra vị trí tương ứng trong ảnh được so khớp
    ### YOUR CODE HERE
    
    # hiển thị kết quả tương ứng
    image = cv2.polylines(image,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

## hiển thị kết quả
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img12 = cv2.drawMatches(template,kp1,image,kp2,good,None,**draw_params)
plt.figure(figsize=(8,8))
plt.imshow(cv2.cvtColor(img12, cv2.COLOR_BGR2RGB))
plt.show()