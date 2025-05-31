import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('CV.png', 0)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened = cv.filter2D(img, -1, kernel)

plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(1, 2, 2), plt.imshow(sharpened, cmap='gray'), plt.title('Sharpened')
plt.show()
