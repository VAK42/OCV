import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('CV.png')
assert img is not None, "File Could Not Be Read - Check With os.path.exists()"

kernel = np.ones((5,5), np.float32) / 25
# Làm Mờ Trung Bình -	np.ones((3,3), np.float32)/9
# Làm Nét	- [[0,-1,0],[-1,5,-1],[0,-1,0]]
# Phát Hiện Cạnh - Sobel: [[1,0,-1],[0,0,0],[-1,0,1]]
# Phát Hiện Cạnh - Laplacian: [[0,1,0],[1,-4,1],[0,1,0]]

dst = cv.filter2D(img, -1, kernel)  # Tích Chập

blur = cv.blur(img, (5,5))
# Mỗi Pixel Mới Là Giá Trị Trung Bình Của Vùng Lân Cận 5x5 Pixel
# Làm Mờ Đều Nhưng Có Thể Làm Mất Chi Tiết & Cạnh Ảnh
gaussian = cv.GaussianBlur(img, (5,5), 0)
# Các Pixel Lân Cận Có Trọng Số Khác Nhau: Pixel Gần Trung Tâm Có Trọng Số Cao Hơn
median = cv.medianBlur(img, 5)
# Mỗi Pixel Được Thay Bằng Giá Trị Trung Vị Trong Vùng 5x5
# Hiệu Quả Rất Cao Trong Việc Loại Bỏ Nhiễu Muối Tiêu
bilateral = cv.bilateralFilter(img, 9, 75, 75)
# Kết Hợp Cả Khoảng Cách Không Gian & Độ Tương Đồng Màu Sắc Để Làm Mờ
# Pixel Xa Hoặc Khác Màu Sẽ Có Ảnh Hưởng Ít Hơn

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
