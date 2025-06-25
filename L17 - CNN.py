import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Tải Dữ Liệu CIFAR-10: 50000 Ảnh Train + 10000 Ảnh Test - Mỗi Ảnh (32, 32, 3)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 2. Resize Lên 64x64 - Chuyển Sang Ảnh Grayscale (1 Kênh) - Chuẩn Hóa [0, 1]
x_train = tf.image.rgb_to_grayscale(tf.image.resize(x_train, [64, 64])) / 255.0
x_test = tf.image.rgb_to_grayscale(tf.image.resize(x_test, [64, 64])) / 255.0

# 3. Định Nghĩa Input Shape Cho Mạng CNN
input_shape = (64, 64, 1)

# 4. Xây Dựng Mô Hình CNN
model = models.Sequential([
    # ----- Block 1 -----
    layers.Conv2D(10, (3, 3), activation='relu', padding='valid', input_shape=input_shape),  
    # Đầu Vào (64, 64, 1) → Conv Với 10 Kernel (3x3x1) → Đầu Ra (62, 62, 10)

    layers.Conv2D(10, (3, 3), activation='relu', padding='valid'),                           
    # (62, 62, 10) → Conv Với 10 Kernel (3x3x10) → Đầu Ra (60, 60, 10)
    # Mỗi Filter Kết Hợp Thông Tin Từ Cả 10 Kênh Đầu Vào

    layers.MaxPooling2D((2, 2)),                                                              
    # (60, 60, 10) → Giảm Kích Thước Còn (30, 30, 10)

    # ----- Block 2 -----
    layers.Conv2D(10, (3, 3), activation='relu', padding='valid'),                           
    # (30, 30, 10) → Conv (3x3x10) → (28, 28, 10)

    layers.Conv2D(10, (3, 3), activation='relu', padding='valid'),                           
    # (28, 28, 10) → Conv → (26, 26, 10)

    layers.MaxPooling2D((2, 2)),                                                              
    # (26, 26, 10) → MaxPool → (13, 13, 10)

    # ----- Flatten + Dense -----
    layers.Flatten(),                                                                         
    # Chuyển Tensor (13, 13, 10) Thành Vector 1D Size 1690

    layers.Dense(10, activation='softmax')                                                    
    # Lớp Fully-Connected Với 10 Output (Cho 10 Class)
    # Trọng Số: Ma Trận Shape (10, 1690) - Mỗi Hàng Là Vector Trọng Số Cho 1 Class
    # Softmax Chuyển Đầu Ra Thành Xác Suất Cho Từng Class
])

# 5. Compile Mô Hình: Dùng Adam + Crossentropy + Theo Dõi Accuracy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 6. In Kiến Trúc Mạng
model.summary()

# 7. Huấn Luyện Mô Hình Trong 10 Epoch Với Batch Size = 64
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=64)

# 8. Đánh Giá Mô Hình Trên Tập Test
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

# 9. Vẽ Biểu Đồ Accuracy & Loss
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.show()

# Weight Dùng Để Xác Định Mức Độ Quan Trọng Của Pixel Đầu Vào Tại Mỗi Vị Trí
# Bias Là Một Số Cộng Vào Kết Quả Sau Khi Tính Tích Chập Để Tăng Tính Linh Hoạt

# H​ = (Input Height − Kernel Height) / 1 + 1
# W = (Input Width − Kernel Width) / 1 + 1
# (Kernel Height × Kernel Width × Input Channels + Bias) × Filters
# Flatten: a x b x c
# Dense: (Input + Bias) x Dense Neurons
# w1​ * x1 ​+ w2 ​* x2 ​+ wmax​ * xmax ​+ b
# Softmax: e**xn / Sum(e**x1 -> e**xmax)

# https://poloclub.github.io/cnn-explainer
# https://www.tensorflow.org/tutorials/images/cnn
# https://colab.research.google.com/drive/1HDyOt73q8nKc1BO1Ol9JmJGo0GDZ-BBY
