import cv2
import argparse
import glob
import numpy as np
import os
from rembg import remove

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def cvt_color_multiple(images):
    cvtImages = []
    if images is not None:
        for image in images:
            cvtImages.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return cvtImages

def show_image_cv2(images,salt):
    i = 0
    if images is not None:
        for image in images:
            imS = cv2.resize(image, (960, 540))  
            cv2.imshow(salt + str(i), imS)
            i += 1
    

def remove_background(images):
    rmvBackgroundImage = []
    if images is not None:
        for image in images:
            new = remove(image)
            rmvBackgroundImage.append(new)
    return rmvBackgroundImage


def canny_edge_detection(images):
    cvtImages = []
    if images is not None:
        for image in images:
            new = cv2.Canny(image, 100, 200)
            cvtImages.append(new)
    return cvtImages
    
def process(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 2)
    img_canny = cv2.Canny(img_blur, 50, 9)
    img_dilate = cv2.dilate(img_canny, np.ones((4, 2)), iterations=11)
    img_erode = cv2.erode(img_dilate, np.ones((13, 7)), iterations=4)
    return cv2.bitwise_not(img_erode)

def get_contours(img):
    contours, _ = cv2.findContours(process(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    cv2.drawContours(img, [cv2.convexHull(cnt)], -1, (0, 0, 255), 2)

ap = argparse.ArgumentParser()
ap.add_argument("-t","--template", required=True, help="Path to template image")
ap.add_argument("-i","--images", required=True, help="Path to images where template will be matched")
ap.add_argument("-v","--visualize", help="Flag indicating whether or not to visualize each iteration")

args = vars(ap.parse_args())

print(args["template"])
templates = load_images_from_folder(args["template"])
templates = remove_background(templates)
#templates = cvt_color_multiple(templates)
#show_image_cv2(templates,"Before")
#templates = canny_edge_detection(templates)
# cv2.imshow("Template",templates[1])

show_image_cv2(templates,"After")

cv2.waitKey(0)
# template = cv2.Canny(template, 50,200)
# (tH, tW) = template.shape[:2]
# cv2.imshow("Template", template)


