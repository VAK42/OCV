import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt

img = cv.imread('CV.png')
assert img is not None, "File Could Not Be Read, Check With os.path.exists()"
px = img[100,100]
print(px)
blue = img[100,100,0]
print(blue)
img[100,100] = [255,255,255]
print(img[100,100])
print(img.shape)
print(img.size)
print(img.dtype)
ball = img[280:340, 330:390]
img[273:333, 100:160] = ball
b,g,r = cv.split(img)
img = cv.merge((b,g,r))
b = img[:,:,0]
img[:,:,2] = 0

BLUE = [255,0,0]
replicate = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_REPLICATE)
reflect = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_REFLECT)
reflect101 = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_REFLECT_101)
wrap = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_WRAP)
constant = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_CONSTANT,value=BLUE)

plt.subplot(231),plt.imshow(img,'gray'),plt.title('Original')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('Replicate')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('Reflect')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('Reflect_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('Wrap')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('Constant')
plt.show()