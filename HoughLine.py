import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread(cv.samples.findFile('CV.png'))
assert img is not None, "File Could Not Be Read - Check With os.path.exists()"

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150, apertureSize=3)
# 50: Ngưỡng Dưới Trong Canny
# 150: Ngưỡng Trên Trong Canny
# apertureSize=3: Kích Thước Kernel Sobel (3×3)

lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
# 1: Độ Phân Giải Khoảng Cách (Pixel)
# np.pi / 180: Độ Phân Giải Góc (π/180 Radian)
# 200: Ngưỡng — Chỉ Lấy Dòng Có Tối Thiểu 200 Điểm

for line in lines:
    rho, theta = line[0]
    # rho: Khoảng Cách Từ Gốc Tọa Độ (0,0) Đến Đường Thẳng
    # theta: Góc Giữa Trục X & Pháp Tuyến Của Đường Thẳng
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

img2 = img.copy()
linesP = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
# 1: Độ Phân Giải Khoảng Cách
# np.pi / 180: Độ Phân Giải Góc
# 100: Ngưỡng Số Bỏ Phiếu Tối Thiểu Để Xác Nhận Dòng
# minLineLength=100: Chiều Dài Dòng Tối Thiểu
# maxLineGap=10: Khoảng Cách Tối Đa Giữa Các Điểm Để Nối Thành Dòng

for line in linesP:
    x1, y1, x2, y2 = line[0]
    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

plt.subplot(121)
plt.imshow(img)
plt.title('Hough Lines Standard')
plt.axis('off')

plt.subplot(122)
plt.imshow(img2)
plt.title('Hough Lines Probabilistic')
plt.axis('off')

plt.show()
