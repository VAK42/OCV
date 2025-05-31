import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('CV.png')
assert img is not None, "File Could Not Be Read, Check With Os.Path.Exists()"

kernel = np.ones((5,5), np.float32) / 25
dst = cv.filter2D(img, -1, kernel)

blur = cv.blur(img, (5,5))
gaussian = cv.GaussianBlur(img, (5,5), 0)
median = cv.medianBlur(img, 5)
bilateral = cv.bilateralFilter(img, 9, 75, 75)

plt.subplot(231), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB)), plt.title('Filter2D Averaging')
plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(cv.cvtColor(blur, cv.COLOR_BGR2RGB)), plt.title('Blur')
plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.imshow(cv.cvtColor(gaussian, cv.COLOR_BGR2RGB)), plt.title('Gaussian Blur')
plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(cv.cvtColor(median, cv.COLOR_BGR2RGB)), plt.title('Median Blur')
plt.xticks([]), plt.yticks([])
plt.subplot(236), plt.imshow(cv.cvtColor(bilateral, cv.COLOR_BGR2RGB)), plt.title('Bilateral Filter')
plt.xticks([]), plt.yticks([])

plt.show()
