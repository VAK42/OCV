import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = tf.image.rgb_to_grayscale(tf.image.resize(x_train, [64, 64])) / 255.0
x_test = tf.image.rgb_to_grayscale(tf.image.resize(x_test, [64, 64])) / 255.0

input_shape = (64, 64, 1)
model = models.Sequential([
    layers.Conv2D(10, (3, 3), activation='relu', padding='valid', input_shape=input_shape),
    layers.Conv2D(10, (3, 3), activation='relu', padding='valid'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(10, (3, 3), activation='relu', padding='valid'),
    layers.Conv2D(10, (3, 3), activation='relu', padding='valid'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test), batch_size=64)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()

# https://poloclub.github.io/cnn-explainer
# https://www.tensorflow.org/tutorials/images/cnn
# https://colab.research.google.com/drive/1HDyOt73q8nKc1BO1Ol9JmJGo0GDZ-BBY
