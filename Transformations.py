import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('CV.png', 0)  # Read Input Image In Grayscale Mode
# cv.IMREAD_COLOR :	1
# cv.IMREAD_GRAYSCALE :	0
# cv.IMREAD_UNCHANGED : -1

c = 255 / np.log(1 + np.max(img))  # 255 / log(1 + max(r))  0 <= r <= 255
logTransformed = c * np.log(1 + img.astype(np.float32))  # s = T(r) = c * log(1 + r)
logTransformed = np.uint8(logTransformed)
# Brighten Images With Low Intensity
# Narrow - Dark Pixel Ranges -> Wider - Brighter
# Compress Bright Pixels Into A Smaller Range

gamma = 2.0
gammaCorrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')
# s = T(r) = c * r ** γ
# γ < 1: Brightens The Image
# γ > 1: Darkens The Image
# Adjust Image Brightness & Contrast

negative = 255 - img  # s = T(r) = 255 - r
# Enhance White & Gray Details In Images With Predominantly Dark Regions

def contrastStretch(img):
    a, b = np.min(img), np.max(img)
    return ((img - a) * 255 / (b - a)).astype('uint8')
stretched = contrastStretch(img)
# Enhance Contrast In Low-Contrast Images

cv.imshow("Original", img)
cv.imshow("Log Transformed", logTransformed)
cv.imshow("Gamma Corrected", gammaCorrected)
cv.imshow("Negative", negative)
cv.imshow("Contrast Stretched", stretched)
cv.waitKey(0)
cv.destroyAllWindows()
