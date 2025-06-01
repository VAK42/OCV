import cv2 as cv
import numpy as np

flags = [i for i in dir(cv) if i.startswith('COLOR_')]  # Tạo Một Danh Sách Gồm Tất Cả Các Thuộc Tính Trong CV Bắt Đầu Bằng 'COLOR_'
print(flags)

cap = cv.VideoCapture(0)  # Mở Camera Mặc Định Để Ghi Hình
while(1):  # Bắt Đầu Một Vòng Lặp Vô Hạn Để Liên Tục Đọc Các Khung Hình
    _, frame = cap.read()  # Đọc Khung Hình Từ Camera & Lưu Vào Frame
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # Chuyển Đổi Khung Hình Từ Không Gian Màu BGR Sang HSV
    lower_blue = np.array([110,50,50])  # Định Nghĩa Ngưỡng Dưới Của Màu Xanh Dương Trong HSV
    upper_blue = np.array([130,255,255])  # Định Nghĩa Ngưỡng Trên Của Màu Xanh Dương Trong HSV
    mask = cv.inRange(hsv, lower_blue, upper_blue)  # Tạo Một Mặt Nạ Chỉ Giữ Lại Những Pixel Nằm Trong Khoảng Màu Xanh Dương
    res = cv.bitwise_and(frame, frame, mask=mask)  # Áp Dụng Mặt Nạ Lên Khung Hình Gốc Để Lọc Ra Vùng Màu Xanh
    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)
    k = cv.waitKey(5) & 0xFF  # Chờ 5 Ms & Lấy Phím Được Nhấn
    if k == 27:  # Nếu Phím ESC Được Nhấn → Thoát Khỏi Vòng Lặp
        break
cv.destroyAllWindows()

green = np.uint8([[[0,255,0]]])  # Tạo Một Pixel Màu Xanh Lá Cây Ở Dạng Mảng NumPy Theo Định Dạng BGR
hsv_green = cv.cvtColor(green, cv.COLOR_BGR2HSV)
print(hsv_green)
