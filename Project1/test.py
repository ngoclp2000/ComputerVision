import cv2
import numpy as np

max_size = (20, 20)

img = cv2.imread('preprocess_img/Chessboard0631.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to remove noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
sharpened = cv2.filter2D(blurred, -1, kernel)

# Get the image dimensions
height, width = img.shape[:2]

# Create a blank canvas with the same number of channels as the input image
canvas = np.zeros((width, height, 3), np.uint8)

# Copy the image data to the canvas, rotated 90 degrees clockwise
canvas = cv2.transpose(sharpened)
canvas = cv2.flip(canvas, 1)

th, im_th_tz = cv2.threshold(canvas, canvas.mean() * 0.975, 255, cv2.THRESH_TOZERO)

result = (0,0) 
ret = None
corners = None

for rows in range(max_size[0]+1,3,-1):
    for cols in range(max_size[1]+1,3,-1):
        if rows > cols :
             continue
        # Try to detect the corners of the chessboard with the current size
        ret, corners = cv2.findChessboardCorners(im_th_tz, (rows, cols), None)
        
        if ret:
            # Chessboard found with the current size, log the size and break from the loops
            result = (rows, cols)
            print("Chessboard size:", result)
            break;
    if ret:
        break
cv2.drawChessboardCorners(img, result, corners, ret)
