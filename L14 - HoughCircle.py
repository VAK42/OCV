import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('CV.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "File Could Not Be Read - Check With os.path.exists()"

img = cv.medianBlur(img, 5)
cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
# Phương Pháp Hough Gradient - Dựa Trên Hướng Gradient Tại Pixel Cạnh
# Tỉ Lệ Giảm Độ Phân Giải: dp=1 ⇒ Dùng Đúng Kích Thước Ảnh Gốc
# Khoảng Cách Tối Thiểu Giữa Tâm Hai Hình Tròn - Giúp Tránh Trùng Lặp
# Ngưỡng Trên Cho Canny Edge Detector - Ngưỡng Dưới Là 50% Tự Động
# Ngưỡng Tích Lũy Vote Tối Thiểu Để Được Coi Là Hình Tròn - Giống Threshold
# Bán Kính Nhỏ Nhất Cần Dò - 0 = Không Giới Hạn
# Bán Kính Lớn Nhất Cần Dò - 0 = Không Giới Hạn

# Làm Tròn Kết Quả & Chuyển Sang Số Nguyên Không Âm Vì Pixel Không Có Số Âm
circles = np.uint16(np.around(circles))

# Vẽ Tất Cả Các Hình Tròn Tìm Được
for i in circles[0, :]:
    # Vẽ Viền Hình Tròn Màu Xanh Lá
    cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # Vẽ Tâm Hình Tròn Màu Đỏ
    cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

cimg = cv.cvtColor(cimg, cv.COLOR_BGR2RGB)

plt.imshow(cimg)
plt.title('Detected Circles')
plt.axis('off')
plt.show()
