import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('CV.png')
assert img is not None, "File Could Not Be Read - Check With os.path.exists()"

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.THRESH_BINARY_INV + cv.THRESH_OTSU: Tự Động Chọn Ngưỡng Otsu - Nhị Phân Đảo (Đen Thành Trắng - Trắng Thành Đen)
Ret, Thresh = cv.threshold(Gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
# Tạo Kernel Hình Vuông 3x3 Để Thực Hiện Phép Toán Hình Thái Học (Morphology)
Kernel = np.ones((3, 3), np.uint8)
# Áp Dụng Phép Mở Để Loại Bỏ Nhiễu Nhỏ Trong Ảnh Nhị Phân
# Opening = Erosion Followed By Dilation
Opening = cv.morphologyEx(Thresh, cv.MORPH_OPEN, Kernel, iterations=2)
# Xác Định Vùng Nền Chắc Chắn (Sure Background) Bằng Phép Giãn (Dilation)
Sure_Bg = cv.dilate(Opening, Kernel, iterations=3)
# Tính Toán Khoảng Cách (Distance Transform) Từ Các Điểm Nền (Background) Đến Biên Của Đối Tượng
Dist_Transform = cv.distanceTransform(Opening, cv.DIST_L2, 5)
# Xác Định Vùng Đối Tượng Chắc Chắn (Sure Foreground) Bằng Cách Lấy Ngưỡng 0.7 Lần Giá Trị Max Trong Khoảng Cách
Ret, Sure_Fg = cv.threshold(Dist_Transform, 0.7 * Dist_Transform.max(), 255, 0)
# Chuyển Sure_Fg Sang Kiểu Uint8 Để Xử Lý Tiếp
Sure_Fg = np.uint8(Sure_Fg)
# Xác Định Vùng Không Xác Định (Unknown) Bằng Cách Lấy Hiệu Giữa Vùng Nền Chắc Chắn & Vùng Đối Tượng Chắc Chắn
Unknown = cv.subtract(Sure_Bg, Sure_Fg)
# Gán Nhãn (Label) Cho Các Vùng Đối Tượng Chắc Chắn (Connected Components)
Ret, Markers = cv.connectedComponents(Sure_Fg)
# Tăng Nhãn Lên 1 Để Tránh Trùng Nhãn Với Nền (Nền: 1)
Markers = Markers + 1
# Gán Vùng Không Xác Định (Unknown) Thành 0 Để Watershed Nhận Diện Đúng Vùng Cần Phân Đoạn
Markers[Unknown == 255] = 0
# Áp Dụng Thuật Toán Watershed Với Ảnh Gốc & Các Nhãn Markers
Markers = cv.watershed(Img, Markers)
# Đánh Dấu Biên Giới Vùng Phân Đoạn (Biên Giới Dc Gán Giá Trị -1) Bằng Màu Đỏ
Img[Markers == -1] = [255, 0, 0]

cv.imshow("Segmented", img)
cv.waitKey(0)
cv.destroyAllWindows()
