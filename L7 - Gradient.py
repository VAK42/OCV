import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('CV.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "File Could Not Be Read - Check With os.path.exists()"

laplacian = cv.Laplacian(img, cv.CV_64F)
# Laplacian - Detect Edges In All Directions
# cv.CV_64F Specifies The Desired Depth Of The Output Image To Prevent Data Loss From Negative Gradients

sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
# 1: Order Of Derivative In X-Direction
# 0: Order Of Derivative In Y-Direction
# ksize=5: Size Of Sobel Kernel (Larger Kernel = Smoother Result)

sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
# 1: Order Of Derivative In X-Direction
# 0: Order Of Derivative In Y-Direction
# ksize=5: Size Of Sobel Kernel (Larger Kernel = Smoother Result)

abs_sobel64f = np.absolute(sobelx)
# Take Absolute Value Of The X-Direction Sobel Result
# Necessary Because Gradient Values Can Be Negative
# -> Avoid Clipping Useful Data When Converting To 8-Bit
# -> Suitable For Visualization

sobel_8u = np.uint8(abs_sobel64f)
# Convert 64-Bit Float To 8-Bit Unsigned Integer
# Useful For Displaying Or Saving The Image As Most Image Formats Expect 8-Bit Values

plt.subplot(2,3,1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X CV_64F'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y CV_64F'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5), plt.imshow(sobelx8u, cmap='gray')
plt.title('Sobel X CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6), plt.imshow(sobel_8u, cmap='gray')
plt.title('Sobel X abs(CV_64F)'), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
