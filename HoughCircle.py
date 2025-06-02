import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('CV.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "File Could Not Be Read - Check With os.path.exists()"

img = cv.medianBlur(img, 5)
# Áp Dụng Bộ Lọc Trung Vị Cho Ảnh Với Kích Thước Mặt Nạ - 5

cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
# Tìm Các Hình Tròn Sử Dụng Biến Đổi Hough - Phương Pháp HOUGH_GRADIENT
# 1: Tỷ Lệ Nghịch Của Độ Phân Giải
# 20: Khoảng Cách Tối Thiểu Giữa Tâm Các Hình Tròn
# param1=50: Ngưỡng Cho Bộ Phát Hiện Cạnh (Canny)
# param2=30: Ngưỡng Cho Giai Đoạn Phát Hiện Trung Tâm
# minRadius=0: Bán Kính Nhỏ Nhất
# maxRadius=0: Bán Kính Lớn Nhất

circles = np.uint16(np.around(circles))
# Làm Tròn Các Giá Trị & Chuyển Sang Dạng Số Nguyên 16 Bit Không Dấu

for i in circles[0, :]:
    cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # Vẽ Hình Tròn Với Tâm (i[0], i[1]) & Bán Kính i[2] - Màu Xanh Lá (0, 255, 0) - Độ Dày 2
    cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    # Vẽ Hình Tròn Với Tâm (i[0], i[1]) & Bán Kính 2 - Màu Đỏ (0, 0, 255) - Độ Dày 3

cimg = cv.cvtColor(cimg, cv.COLOR_BGR2RGB)

plt.imshow(cimg)
plt.title('Detected Circles')
plt.axis('off')
plt.show()
