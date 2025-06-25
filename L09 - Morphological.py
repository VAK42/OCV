import cv2 as cv
import numpy as np

img = cv.imread('CV.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "File Could Not Be Read - Check With os.path.exists()"

# Create Structuring Elements (Kernels) Of Size 5x5 In Different Shapes

# Rectangular Kernel
kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
# Elliptical Kernel
kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
# Cross-Shaped Kernel
kernel_cross = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))

# -------------------- Using Rectangular Kernel --------------------

# Erosion Removes White Noise Or Shrinks Objects
erosion_rect = cv.erode(img, kernel_rect, iterations=1)
cv.imshow("Erosion - Rectangular", erosion_rect)

# Dilation Expands White Areas Or Grows Objects
dilation_rect = cv.dilate(img, kernel_rect, iterations=1)
cv.imshow("Dilation - Rectangular", dilation_rect)

# Opening = Erosion Followed By Dilation - Removes Small White Noise
opening_rect = cv.morphologyEx(img, cv.MORPH_OPEN, kernel_rect)
cv.imshow("Opening - Rectangular", opening_rect)

# Closing = Dilation Followed By Erosion - Fills Small Black Holes
closing_rect = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_rect)
cv.imshow("Closing - Rectangular", closing_rect)

# Morphological Gradient = Difference Between Dilation & Erosion
gradient_rect = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel_rect)
cv.imshow("Gradient - Rectangular", gradient_rect)

# Top Hat = Original Image - Opening <-> Highlights Small Bright Objects
tophat_rect = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel_rect)
cv.imshow("Tophat - Rectangular", tophat_rect)

# Black Hat = Closing - Original Image <-> Highlights Small Dark Spots
blackhat_rect = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel_rect)
cv.imshow("Blackhat - Rectangular", blackhat_rect)

# -------------------- Using Elliptical Kernel --------------------

erosion_ellipse = cv.erode(img, kernel_ellipse, iterations=1)
cv.imshow("Erosion - Elliptical", erosion_ellipse)
dilation_ellipse = cv.dilate(img, kernel_ellipse, iterations=1)
cv.imshow("Dilation - Elliptical", dilation_ellipse)
opening_ellipse = cv.morphologyEx(img, cv.MORPH_OPEN, kernel_ellipse)
cv.imshow("Opening - Elliptical", opening_ellipse)
closing_ellipse = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_ellipse)
cv.imshow("Closing - Elliptical", closing_ellipse)
gradient_ellipse = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel_ellipse)
cv.imshow("Gradient - Elliptical", gradient_ellipse)
tophat_ellipse = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel_ellipse)
cv.imshow("Tophat - Elliptical", tophat_ellipse)
blackhat_ellipse = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel_ellipse)
cv.imshow("Blackhat - Elliptical", blackhat_ellipse)

# -------------------- Using Cross-Shaped Kernel --------------------

erosion_cross = cv.erode(img, kernel_cross, iterations=1)
cv.imshow("Erosion - Cross", erosion_cross)
dilation_cross = cv.dilate(img, kernel_cross, iterations=1)
cv.imshow("Dilation - Cross", dilation_cross)
opening_cross = cv.morphologyEx(img, cv.MORPH_OPEN, kernel_cross)
cv.imshow("Opening - Cross", opening_cross)
closing_cross = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_cross)
cv.imshow("Closing - Cross", closing_cross)
gradient_cross = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel_cross)
cv.imshow("Gradient - Cross", gradient_cross)
tophat_cross = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel_cross)
cv.imshow("Tophat - Cross", tophat_cross)
blackhat_cross = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel_cross)
cv.imshow("Blackhat - Cross", blackhat_cross)
cv.waitKey(0)
cv.destroyAllWindows()
