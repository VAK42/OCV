import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('CV.png', 0)  # Read Input Image In Grayscale Mode
# cv.IMREAD_COLOR :	1
# cv.IMREAD_GRAYSCALE :	0
# cv.IMREAD_UNCHANGED : -1
c = 255 / np.log(1 + np.max(img))
log_transformed = c * np.log(1 + img.astype(np.float32))
log_transformed = np.uint8(log_transformed)
gamma = 2.0
gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')
negative = 255 - img
def contrastStretch(img):
    a, b = np.min(img), np.max(img)
    return ((img - a) * 255 / (b - a)).astype('uint8')
stretched = contrastStretch(img)
cv.imshow("Original", img)
cv.imshow("Log Transformed", log_transformed)
cv.imshow("Gamma Corrected", gamma_corrected)
cv.imshow("Negative", negative)
cv.imshow("Contrast Stretched", stretched)
cv.waitKey(0)
cv.destroyAllWindows()
