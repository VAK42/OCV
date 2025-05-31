import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('CV.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# Of Pixels')
plt.plot(gray_hist)
plt.xlim([0, 256])
plt.show()

colors = ('b', 'g', 'r')
plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# Of Pixels')
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.show()

equalized = cv.equalizeHist(gray)
cv.imshow('Original Grayscale', gray)
cv.imshow('Equalized Grayscale', equalized)
cv.waitKey(0)
cv.destroyAllWindows()

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray)
cv.imshow('CLAHE Grayscale', clahe_img)
cv.waitKey(0)
cv.destroyAllWindows()

mask = np.zeros(gray.shape[:2], np.uint8)
mask[100:300, 100:300] = 255
masked_img = cv.bitwise_and(gray, gray, mask=mask)
masked_hist = cv.calcHist([gray], [0], mask, [256], [0, 256])
plt.figure()
plt.title('Masked Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# Of Pixels')
plt.plot(masked_hist)
plt.xlim([0, 256])
plt.show()

roi = img[100:200, 100:200]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
roi_hist = cv.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
back_proj = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
cv.imshow('BackProjection', back_proj)
cv.waitKey(0)
cv.destroyAllWindows()
