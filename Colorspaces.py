import cv2 as cv
import numpy as np

flags = [i for i in dir(cv) if i.startswith('COLOR_')]  # Create A List Of All Attributes In CV That Start With 'COLOR_'
print(flags)

cap = cv.VideoCapture(0)  # Open The Default Camera For Video Capture
while(1):  # Start An Infinite Loop To Continuously Read Frames
    _, frame = cap.read()  # Read A Frame From The Camera And Store It In Frame
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # Convert The Frame From BGR Colorspace To HSV Colorspace
    lower_blue = np.array([110,50,50])  # Define The Lower Bound Of Blue Color In HSV
    upper_blue = np.array([130,255,255])  # Define The Upper Bound Of Blue Color In HSV
    mask = cv.inRange(hsv, lower_blue, upper_blue)  # Create A Mask That Keeps Only Pixels Within The Blue Range
    res = cv.bitwise_and(frame, frame, mask=mask)  # Apply The Mask To The Original Frame To Extract Blue Areas
    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)
    k = cv.waitKey(5) & 0xFF  # Wait For 5 Milliseconds & Get The Pressed Key
    if k == 27:  # If Escape Key Is Pressed -> Exit The Loop
        break
cv.destroyAllWindows()

green = np.uint8([[[0,255,0]]])  # Create A Green Pixel In BGR Format As A NumPy Array
hsv_green = cv.cvtColor(green, cv.COLOR_BGR2HSV)
print(hsv_green)
