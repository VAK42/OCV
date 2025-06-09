import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('CV.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "File Could Not Be Read - Check With os.path.exists()"

edges = cv.Canny(img, 100, 200)
# Phát Hiện Các Biên Thực Sự Trong Ảnh - Đồng Thời Loại Bỏ Nhiễu -> Tạo Ra Các Đường Biên Sắc Nét + Chính Xác + Liên Tục
# I - Giảm Nhiễu Bằng Gaussian Blur
# + Ảnh Gốc Thường Chứa Nhiễu & Các Biến Động Nhỏ Có Thể Bị Nhầm Là Biên → Cần Làm Mượt Ảnh Trước
# + Với Mỗi Pixel:
# + Nhân Từng Điểm Với Trọng Số Tương Ứng Trong Kernel
# + Cộng Tất Cả Lại Rồi Chia Tổng Trọng Số → Ra Giá Trị Mới

# II - Tính Biên Độ & Hướng Gradient
# + Tìm Những Vùng Có Sự Thay Đổi Cường Độ Lớn → Có Thể Là Biên
# + Sobel 3x3
# Gx = [-1, 0, +1]
#      [-2, 0, +2]
#      [-1, 0, +1]
# Gy = [-1, -2, -1]
#      [ 0,  0,  0]
#      [+1, +2, +1]
# + Áp Dụng Tích Chập Dùng Bộ Lọc Sobel Vào Mảng Ảnh
# + Gx: Tính Tổng Mảng Ảnh Sau Khi Áp Dụng Bộ Lộc Sobel Theo Hướng Ngang
# + Gy: Tính Tổng Mảng Ảnh Sau Khi Áp Dụng Bộ Lộc Sobel Theo Hướng Dọc
# + Độ Lớn Gradient: sqrt(Gx ** 2 + Gy ** 2) → Càng Lớn → Càng Có Khả Năng Là Biên
# + Hướng Gradient: arctan(Gy/Gx)

# III - Làm Mảnh Biên
# + Làm Cho Biên Mảnh Lại Bằng Cách Giữ Lại Những Pixel Có Giá Trị Lớn Nhất Theo Hướng Gradient
# + Với Mỗi Pixel - Ta So Sánh Giá Trị Gradient Của Nó Với Hai Điểm Lân Cận Theo Hướng Gradient
# + Nếu Không Phải Giá Trị Lớn Nhất → Loại Bỏ

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
