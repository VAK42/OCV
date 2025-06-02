import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('CV.png')
assert img is not None, "File Could Not Be Read - Check With os.path.exists()"

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

Ret, Thresh = cv.threshold(Gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
# Ngưỡng Hóa Otsu Kèm Phép Đảo Nhị Phân

Kernel = np.ones((3, 3), np.uint8)
# Tạo Kernel Ma Trận 3x3 Toàn Số 1 Dạng Uint8

Opening = cv.morphologyEx(Thresh, cv.MORPH_OPEN, Kernel, iterations=2)
# Phép Mở (Làm Mòn → Giãn Nở) Để Loại Bỏ Nhiễu Nhỏ - Lặp 2 Lần

Sure_Bg = cv.dilate(Opening, Kernel, iterations=3)
# Giãn Nở Để Tính Vùng Nền Chắc Chắn (Sure Background)

Dist_Transform = cv.distanceTransform(Opening, cv.DIST_L2, 5)
# Tính Toạ Độ Khoảng Cách Từ Mỗi Điểm Đến Vùng Nền (Distance Transform)

Ret, Sure_Fg = cv.threshold(Dist_Transform, 0.7 * Dist_Transform.max(), 255, 0)
# Ngưỡng Hóa Khoảng Cách Để Lấy Vùng Tiêu Chuẩn Chắc Chắn (Sure Foreground)

Sure_Fg = np.uint8(Sure_Fg)
# Chuyển Đổi Dạng Ảnh Về Uint8 Để Xử Lý Tiếp

Unknown = cv.subtract(Sure_Bg, Sure_Fg)
# Tính Vùng Chưa Biết (Unknown) Bằng Phép Trừ Vùng Nền Với Vùng Tiêu Chuẩn

Ret, Markers = cv.connectedComponents(Sure_Fg)
# Đánh Dấu Các Vùng Kết Nối Trong Vùng Tiêu Chuẩn Chắc Chắn

Markers = Markers + 1
# Tăng Giá Trị Các Marker Lên 1 Để Phân Biệt Với Nền

Markers[Unknown == 255] = 0
# Gán Vùng Chưa Biết Là 0 Để Phân Biệt Trong Quá Trình Watershed

Markers = cv.watershed(Img, Markers)
# Áp Dụng Thuật Toán Watershed Để Phân Vùng

Img[Markers == -1] = [255, 0, 0]
# Tô Màu Biên Vùng Phân Vùng Bằng Màu Đỏ (BGR)

cv.imshow("Segmented", img)
cv.waitKey(0)
cv.destroyAllWindows()
