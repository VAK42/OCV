import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('CV.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "File Could Not Be Read - Check With os.path.exists()"

laplacian = cv.Laplacian(img, cv.CV_64F)
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
abs_sobel64f = np.absolute(sobelx)
sobel_8u = np.uint8(abs_sobel64f)

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
