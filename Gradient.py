import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('CV.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "File Could Not Be Read - Check With os.path.exists()"

laplacian = cv.Laplacian(img, cv.CV_64F)
# Tính Đạo Hàm Bậc Hai Của Ảnh Để Phát Hiện Các Vùng Có Biên Rõ
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
# Tính Đạo Hàm Bậc Nhất Theo Trục X
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
# Tính Đạo Hàm Bậc Nhất Theo Trục Y
abs_sobel64f = np.absolute(sobelx)
# Lấy Giá Trị Tuyệt Đối Của Gradient Để Dễ Biểu Diễn Trên Ảnh
sobel_8u = np.uint8(abs_sobel64f)
# Ép Kết Quả Về Kiểu uint8 (0–255) Để Có Thể Hiển Thị Trên Ảnh

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
