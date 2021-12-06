import cv2
import numpy as np

# a.Allow to load image from image gallery
image = cv2.imread('OpenCV_Assignment_Image.png', cv2.IMREAD_COLOR)
ROI = image[50:350, 300:650]
gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.blur(gray, (3, 3))

# b. Identifying the circle objects (wood log) from the image(using Hough transform)
detected_circles = cv2.HoughCircles(gray_blurred,
                                    cv2.HOUGH_GRADIENT, minDist=12, dp=30, param1=50,
                                    param2=30, minRadius=20, maxRadius=30)  # 12,30,50,30,20,30

# c. Draw circles that are detected
if detected_circles is not None:
    detected_circles = np.uint16(np.around(detected_circles))
    count = 0
    for i in detected_circles[0, :]:
        a, b, r = i[0], i[1], i[2]
        cv2.circle(ROI, (a, b), r, (0, 255, 0), 2)
        count = count + 1
    # d. Identify the number of circle objects
    print('Number of circles detected:', count)
    cv2.imshow("Output_Image", image)
    cv2.waitKey(0)
