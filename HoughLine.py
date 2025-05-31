import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread(cv.samples.findFile('CV.png'))
assert img is not None, "File Could Not Be Read - Check With os.path.exists()"

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

edges = cv.Canny(gray, 50, 150, apertureSize=3)

img1 = img.copy()
lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 2)

img2 = img.copy()
linesP = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
for line in linesP:
    x1, y1, x2, y2 = line[0]
    cv.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)

img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

plt.subplot(121)
plt.imshow(img1)
plt.title('Hough Lines Standard')
plt.axis('off')

plt.subplot(122)
plt.imshow(img2)
plt.title('Hough Lines Probabilistic')
plt.axis('off')

plt.show()
