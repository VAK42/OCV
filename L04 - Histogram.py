import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('CV.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])
# - [0]: Channel Index (0: Grayscale)
# - None: No Mask - Use Full Image
# - [256]: Number Of Bins
# - [0, 256]: Intensity Range
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
    # - [i]: Channel Index (0: Blue | 1: Green | 2: Red)
    # - None: No Mask - Use Full Image
    # - [256]: Number Of Bins
    # - [0, 256]: Intensity Range
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.show()

equalized = cv.equalizeHist(gray)
# Spread Out Pixel Intensities To Improve Contrast
# Only Work On Single-channel - Grayscale
cv.imshow('Original Grayscale', gray)
cv.imshow('Equalized Grayscale', equalized)
cv.waitKey(0)
cv.destroyAllWindows()
