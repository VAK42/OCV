import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('CV.png', 0)
c = 255 / np.log(1 + np.max(img))
logTransformed = c * np.log(1 + img.astype(np.float32))
logTransformed = np.uint8(logTransformed)
gamma = 2.0
gammaCorrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')
negative = 255 - img

def contrastStretch(img):
    a, b = np.min(img), np.max(img)
    return ((img - a) * 255 / (b - a)).astype('uint8')
stretched = contrastStretch(img)

cv.imshow("Original", img)
cv.imshow("Log Transformed", logTransformed)
cv.imshow("Gamma Corrected", gammaCorrected)
cv.imshow("Negative", negative)
cv.imshow("Contrast Stretched", stretched)
cv.waitKey(0)
cv.destroyAllWindows()
