import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('CV.png')
assert img is not None, "File Could Not Be Read - Check With os.path.exists()"

kernel = np.ones((5,5), np.float32) / 25
# [1/25, 1/25, 1/25, 1/25, 1/25]
# [1/25, 1/25, 1/25, 1/25, 1/25]
# [1/25, 1/25, 1/25, 1/25, 1/25]
# [1/25, 1/25, 1/25, 1/25, 1/25]
# [1/25, 1/25, 1/25, 1/25, 1/25]

dst = cv.filter2D(img, -1, kernel)
# Convolution
# Smooth The Image By Averaging The Surrounding Pixels

blur = cv.blur(img, (5,5))
# Basic Smoothing - Can Remove Minor Noise But Also Blur Edges

gaussian = cv.GaussianBlur(img, (5,5), 0)
# Size Of The Gaussian Kernel Must Be Odd
# 0: Standard Deviation In X Direction
# Better Smoothing Than Blur - Weight Center Pixels More
# Preserve Edges Slightly Better Than Average Blur

median = cv.medianBlur(img, 5)
# 5: Aperture Size Must Be Odd & > 1
# Excellent For Removing Salt-Pepper Noise
# Preserve Edges Much Better Than Averaging Filters

bilateral = cv.bilateralFilter(img, 9, 75, 75)
# 9: Diameter Of Pixel Neighborhood
# 75: sigmaColor – How Much Color Difference Is Tolerated
# 75: sigmaSpace – How Far In Space To Search For Neighbors
# Smooth Flat Areas While Preserving Sharp Edges
# Useful In Tasks: Face Beautification - Cartoon Effects

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
