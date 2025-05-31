import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt

img = cv.imread('CV.png')
assert img is not None, "File Could Not Be Read - Check With os.path.exists()"
px = img[100,100]  # Access The Pixel Value At Row 100 & Column 100
print(px)
blue = img[100,100,0]  # Access The Blue Channel Value Of The Pixel At (100,100) - 0
green = img[100,100,1]  # Access The Green Channel Value Of The Pixel At (100,100) - 1
red = img[100,100,2]  # Access The Red Channel Value Of The Pixel At (100,100) - 2
print(blue)
print(green)
print(red)
img[100,100] = [255,255,255]  # Set The Pixel At (100,100) To White
print(img[100,100])
print(img.shape)  # (Height, Width, Channels)
print(img.size)  # Height × Width × Channels
print(img.dtype)
ball = img[280:340, 330:390]
# 280:340 → Rows: Start At 280 - Go Up To 339 → Total Of 60 Rows
# 330:390 → Columns: Start At 330 - Go Up To 389 → Total Of 60 Columns
img[273:333, 100:160] = ball
b,g,r = cv.split(img)  # Split The Image Into Blue - Green - Red Channels
img = cv.merge((b,g,r))  # Merge The Blue - Green - Red Channels Back Into Image
b = img[:,:,0]  # Extract Only The Blue Channel From The Image
img[:,:,2] = 0  # Set All Red Channel Values To Zero (Remove Red Color)

BLUE = [255,0,0]
replicate = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_REPLICATE)  # Add A Border By Replicating Edge Pixels
reflect = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_REFLECT)  # Add A Border By Reflecting Pixels At The Edge
reflect101 = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_REFLECT_101)  # Add A Border By Reflecting But Skipping The Last Row/Column
wrap = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_WRAP)  # Add A Border By Wrapping Pixels Around From Opposite Edges
constant = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_CONSTANT,value=BLUE)  # Add A Constant Blue Border Around The Image

plt.subplot(231),plt.imshow(img,'gray'),plt.title('Original')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('Replicate')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('Reflect')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('Reflect_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('Wrap')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('Constant')
plt.show()
