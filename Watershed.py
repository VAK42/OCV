import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('CV.png')
assert img is not None, "File Could Not Be Read - Check With os.path.exists()"

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
Ret, Thresh = cv.threshold(Gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
Kernel = np.ones((3, 3), np.uint8)
Opening = cv.morphologyEx(Thresh, cv.MORPH_OPEN, Kernel, iterations=2)
Sure_Bg = cv.dilate(Opening, Kernel, iterations=3)
Dist_Transform = cv.distanceTransform(Opening, cv.DIST_L2, 5)
Ret, Sure_Fg = cv.threshold(Dist_Transform, 0.7 * Dist_Transform.max(), 255, 0)
Sure_Fg = np.uint8(Sure_Fg)
Unknown = cv.subtract(Sure_Bg, Sure_Fg)
Ret, Markers = cv.connectedComponents(Sure_Fg)
Markers = Markers + 1
Markers[Unknown == 255] = 0
Markers = cv.watershed(Img, Markers)
Img[Markers == -1] = [255, 0, 0]

cv.imshow("Segmented", img)
cv.waitKey(0)
cv.destroyAllWindows()
