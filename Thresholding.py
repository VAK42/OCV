import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('CV.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "File Could Not Be Read - Check With os.path.exists()"

medianBlur = cv.medianBlur(img, 5)
# Lọc Trung Vị ĐỂ GIẢM NHIỄU MUỐI TIÊU
gaussianBlur = cv.GaussianBlur(img, (5, 5), 0)
# Lọc Gaussian ĐỂ LÀM MỜ ẢNH MỀM MẠI HƠN

ret, t1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
# Ngưỡng Hóa Toàn Cục - NHỊ PHÂN BÌNH THƯỜNG
ret, t2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
# Ngưỡng Hóa Toàn Cục - NHỊ PHÂN ĐẢO NGƯỢC
ret, t3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
# Ngưỡng Hóa Toàn Cục - CẮT NGƯỠNG (TRUNC)
ret, t4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
# Ngưỡng Hóa Toàn Cục - GIỮ GIÁ TRỊ KHI VƯỢT NGƯỠNG (TOZERO)
ret, t5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)
# Ngưỡng Hóa Toàn Cục - ĐẢO GIÁ TRỊ TOZERO

t6 = cv.adaptiveThreshold(medianBlur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
# Ngưỡng Hóa Thích Nghi - PHƯƠNG PHÁP TRUNG BÌNH CỘNG
t7 = cv.adaptiveThreshold(medianBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
# Ngưỡng Hóa Thích Nghi - PHƯƠNG PHÁP GAUSSIAN

ret, t8 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# Ngưỡng Hóa OTSU TRÊN ẢNH GỐC
ret, t9 = cv.threshold(gaussianBlur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# Ngưỡng Hóa OTSU SAU KHI LỌC GAUSSIAN

# TÍNH HISTOGRAM CỦA ẢNH GAUSSIAN ĐỂ DÙNG TRONG TÍNH TOÁN OTSU THỦ CÔNG
hist = cv.calcHist([gaussian_blur], [0], None, [256], [0, 256])
hist_norm = hist.ravel() / hist.sum()
Q = hist_norm.cumsum()
bins = np.arange(256)
fn_min = np.inf
manual_thresh = -1

# VÒNG LẶP TÍNH TOÁN NGƯỠNG TỐI ƯU THEO OTSU THỦ CÔNG
for i in range(1, 256):
    p1, p2 = np.hsplit(hist_norm, [i])
    q1, q2 = Q[i], Q[255] - Q[i]
    if q1 < 1.e-6 or q2 < 1.e-6:
        continue
    b1, b2 = np.hsplit(bins, [i])
    m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
    v1 = np.sum(((b1 - m1) ** 2) * p1) / q1
    v2 = np.sum(((b2 - m2) ** 2) * p2) / q2
    fn = v1 * q1 + v2 * q2
    if fn < fn_min:
        fn_min = fn
        manual_thresh = i

titles = ['Original', 'Binary', 'Binary Inv', 'Trunc', 'ToZero', 'ToZero Inv', 'Adaptive Mean', 'Adaptive Gauss', 'Global Otsu', 'Otsu + Gaussian', 'Histogram']
images = [img, t1, t2, t3, t4, t5, t6, t7, t8, t9]

plt.figure(figsize=(16, 10))

for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.subplot(3, 4, 11)
plt.hist(gaussian_blur.ravel(), 256)
plt.axvline(x=manual_thresh, color='r', linestyle='--')
plt.title(titles[10]), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
