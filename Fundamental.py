import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt

img = cv.imread('CV.png')
assert img is not None, "File Could Not Be Read - Check With os.path.exists()"
px = img[100,100]  # Truy Cập Giá Trị Pixel Tại Dòng 100 Và Cột 100
print(px)
blue = img[100,100,0]  # Truy Cập Giá Trị Kênh Màu Xanh Dương Của Pixel Tại (100,100) - 0
green = img[100,100,1]  # Truy Cập Giá Trị Kênh Màu Xanh Lá Của Pixel Tại (100,100) - 1
red = img[100,100,2]  # Truy Cập Giá Trị Kênh Màu Đỏ Của Pixel Tại (100,100) - 2
print(blue)
print(green)
print(red)
img[100,100] = [255,255,255]  # Gán Pixel Tại (100,100) Thành Trắng
print(img[100,100])
print(img.shape)  # (Height, Width, Channel)
print(img.size)  # Height × Width × Channel
print(img.dtype)
ball = img[280:340, 330:390]
# 280:340 → Dòng: Bắt Đầu Từ 280 - Đến 339 → Tổng Là 60 Dòng
# 330:390 → Cột: Bắt Đầu Từ 330 - Đến 389 → Tổng Là 60 Cột
img[273:333, 100:160] = ball
b,g,r = cv.split(img)  # Tách Ảnh Thành Các Kênh Xanh Dương - Xanh Lá - Đỏ
img = cv.merge((b,g,r))  # Gộp Các Kênh Xanh Dương - Xanh Lá - Đỏ Trở Lại Thành Ảnh
b = img[:,:,0]  # Trích Xuất Chỉ Kênh Màu Xanh Dương Từ Ảnh
img[:,:,2] = 0  # Gán Tất Cả Giá Trị Kênh Đỏ Thành 0 (Loại Bỏ Màu Đỏ)

BLUE = [255,0,0]
replicate = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_REPLICATE)  # Thêm Viền Bằng Cách Lặp Lại Pixel Cạnh
reflect = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_REFLECT)  # Thêm Viền Bằng Cách Phản Chiếu Pixel Ở Mép
reflect101 = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_REFLECT_101)  # Thêm Viền Bằng Cách Phản Chiếu Nhưng Bỏ Qua Hàng/Cột Cuối
wrap = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_WRAP)  # Thêm Viền Bằng Cách Cuộn Pixel Từ Mép Đối Diện
constant = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_CONSTANT,value=BLUE)  # Thêm Viền Màu Xanh Dương Cố Định Quanh Ảnh

plt.subplot(231),plt.imshow(img,'gray'),plt.title('Original')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('Replicate')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('Reflect')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('Reflect_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('Wrap')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('Constant')
plt.show()
