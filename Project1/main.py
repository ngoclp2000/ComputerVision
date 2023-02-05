
import cv2
import numpy as np

# Load the image
img = cv2.imread('./images/Chessboard0631.png')

# Resize the image
img = cv2.resize(img, (960, 720))

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply image thresholding
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Apply image denoising
binary = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)

# Enhance image contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
binary = clahe.apply(binary)

# Segment the image into regions of interest
segmented = cv2.pyrMeanShiftFiltering(binary, sp=20, sr=20)

# Define the chessboard size
chessboard_size = (9, 6)

# Find the corners of the chessboard pattern
found, corners = cv2.findChessboardCorners(segmented, chessboard_size, None)

if found:
    # Calculate the size of the chessboard squares
    square_size = np.linalg.norm(corners[0] - corners[chessboard_size[0]])

    # Draw the corners on the image
    cv2.drawChessboardCorners(img, chessboard_size, corners, found)

    # Show the image with the corners drawn
    cv2.imshow('Chessboard corners', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print('Chessboard not found')
