import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('CV.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "File Could Not Be Read - Check With os.path.exists()"

# Thresholding: Separate Objects From The Background In A Grayscale Image
# Pixel's Intensity > Certain Value ? Foreground : Background
# Simple Thresholding	- Uniform Lighting	 - Apply 1 Global Threshold Value
# Adaptive Thresholding	- Uneven Lighting	 - Threshold Value Is Computed For Small Regions
# Otsu’s Thresholding	- Bimodal Histograms - Automatically Find Best Threshold By Minimizing Intra-class Variance

medianBlur = cv.medianBlur(img, 5)
# Reduce Salt-And-Pepper Noise

gaussianBlur = cv.GaussianBlur(img, (5, 5), 0)
# Smooth The Image

# Apply Simple Global Thresholding
ret, t1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
# Binary Threshold: If Pixel > 127 - Set To 255 - Else 0
ret, t2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
# Inverse Binary Threshold: If Pixel > 127 - Set To 0 - Else 255
ret, t3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
# Truncate Threshold: If Pixel > 127 - Set To 127 - Else Keep Original
ret, t4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
# ToZero Threshold: If Pixel > 127 - Keep Original - Else Set To 0
ret, t5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)
# Inverse ToZero Threshold: If Pixel > 127 - Set To 0 - Else Keep Original

# Apply Adaptive Thresholding - Local Thresholding Per Pixel Neighborhood
t6 = cv.adaptiveThreshold(medianBlur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
# Adaptive Mean: Use Mean Of Neighborhood (Block Size 11) - Constant 2
t7 = cv.adaptiveThreshold(medianBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
# Adaptive Gaussian: Use Weighted Sum (Gaussian Window) - Constant 2
ret, t8 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# Otsu's Thresholding: Automatically Find Best Threshold To Separate Foreground/Background
ret, t9 = cv.threshold(gaussianBlur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# Otsu's Thresholding After Gaussian Blur: Usually Produces Better Results On Noisy Images
hist = cv.calcHist([gaussian_blur], [0], None, [256], [0, 256])
# Calculate Histogram Of Blurred Image
hist_norm = hist.ravel() / hist.sum()
# Normalize Histogram To Get Probability Distribution
Q = hist_norm.cumsum()
# Compute Cumulative Sum Of Normalized Histogram
bins = np.arange(256)
# Generate Array Of Bin Values (0 To 255)
fn_min = np.inf
manual_thresh = -1
# Initialize Minimum Within-Class Variance (Fn) To Infinity

# Loop Through All Threshold Values (1 To 255) To Find Optimal Threshold Manually - Otsu's Idea
for i in range(1, 256):
    # Split Histogram Into Two Classes At Index i
    p1, p2 = np.hsplit(hist_norm, [i])  # Probabilities
    q1, q2 = Q[i], Q[255] - Q[i]        # Cumulative Sums
    # Skip If Class Probability Too Small To Avoid Division Errors
    if q1 < 1.e-6 or q2 < 1.e-6:
        continue
    # Split Bin Values
    b1, b2 = np.hsplit(bins, [i])
    # Calculate Class Means
    m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
    # Calculate Class Variances
    v1 = np.sum(((b1 - m1) ** 2) * p1) / q1
    v2 = np.sum(((b2 - m2) ** 2) * p2) / q2
    # Compute Weighted Within-Class Variance
    fn = v1 * q1 + v2 * q2
    # Update Minimum Variance & Threshold
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
