import cv2 
import numpy as np

img = cv2.imread("./templates/stuffed_animal_1.jpg", cv2.IMREAD_GRAYSCALE) 
frame = cv2.imread("./images/20221206_003418.jpg", cv2.IMREAD_GRAYSCALE)

# if SIFT_create() gives problems, try downgrading opencv with
# pip uninstall opencv-python
# pip install opencv-contrib-python==3.4.2.17
sift = cv2.xfeatures2d.SIFT_create() 
kp_image, desc_image = sift.detectAndCompute(img, None) 
kp_frame, desc_frame = sift.detectAndCompute(frame, None) 

index_params = dict(algorithm=0, trees=5) 
search_params = dict() 
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc_image, desc_frame, k=2)

# clean the matches
good_points=[] 
for m, n in matches: 
    if(m.distance < 0.6 * n.distance): 
        good_points.append(m)

query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2) 
train_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

# find homography to find mask
matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0) 
matches_mask = mask.ravel().tolist()
h,w = img.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts, matrix)
homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3) 

cv2.imshow(homography) 