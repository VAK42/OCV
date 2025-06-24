import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('CV.png', 0)
# 0: Grayscale
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# Define A Sharpening Kernel - 3x3 Matrix
# Enhance The Center Pixel & Subtract The Neighboring Ones
# Sharpen The Image By Emphasizing Edges & Details
sharpened = cv.filter2D(img, -1, kernel)
# Apply The Sharpening Kernel To The Image Using 2D Convolution
# -1: The Output Image Will Have The Same Depth As The Input Image

plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(1, 2, 2), plt.imshow(sharpened, cmap='gray'), plt.title('Sharpened')
plt.show()
