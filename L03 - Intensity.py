import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('CV.png', 0)
# 0: Grayscale

c = 255 / np.log(1 + np.max(img))
# LOGARITHMIC TRANSFORMATION
# Compute Scaling Constant Based On Max Pixel Value - 255

logTransformed = c * np.log(1 + img.astype(np.float32))
# Apply Log Transform: s = c * log(1 + r)
logTransformed = np.uint8(logTransformed)

gamma = 2.0
# GAMMA CORRECTION
# Gamma > 1 - Darken
# Gamma < 1 - Brighten
gammaCorrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')

negative = 255 - img
# NEGATIVE TRANSFORMATION
# Simple Inversion: s = 255 - r
# Create A Photographic Negative Of The Image

def contrastStretch(img):
    a, b = np.min(img), np.max(img)
    # s = (r - a) * 255 / (b - a)
    return ((img - a) * 255 / (b - a)).astype('uint8')
stretched = contrastStretch(img)
# CONTRAST STRETCHING - LINEAR TRANSFORMATION

cv.imshow("Original", img)
cv.imshow("Log Transformed", logTransformed)
cv.imshow("Gamma Corrected", gammaCorrected)
cv.imshow("Negative", negative)
cv.imshow("Contrast Stretched", stretched)
cv.waitKey(0)
cv.destroyAllWindows()
