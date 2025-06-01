import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('CV.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "File Could Not Be Read - Check With os.path.exists()"

edges = cv.Canny(img, 100, 200)
# Phát Hiện Các Biên Thực Sự Trong Ảnh - Đồng Thời Loại Bỏ Nhiễu -> Tạo Ra Các Đường Biên Sắc Nét + Chính Xác + Liên Tục
# I - Giảm Nhiễu Bằng Gaussian Blur
# + Ảnh Gốc Thường Chứa Nhiễu & Các Biến Động Nhỏ Có Thể Bị Nhầm Là Biên → Cần Làm Mượt Ảnh Trước
# II - Tính Biên Độ & Hướng Gradient
# + Tìm Những Vùng Có Sự Thay Đổi Cường Độ Lớn → Có Thể Là Biên
# III - Làm Mảnh Biên
# + Làm Cho Biên Mảnh Lại Bằng Cách Giữ Lại Những Pixel Có Giá Trị Lớn Nhất Theo Hướng Gradient
# IV - Ngưỡng Kép
# + Biên Mạnh → Chắc Chắn Là Biên - G >= 200
# + Biên Yếu → Có Thể Là Biên - 100 <= G < 200
# + Không Phải Biên → Loại Bỏ - G < 100
# V - Theo Dõi Cạnh
# + Giữ Lại Pixel Biên Yếu Nếu Nó Liên Kết Với Biên Mạnh & Ngược Lại Thì Bỏ

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
