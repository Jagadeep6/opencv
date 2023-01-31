import pytesseract as tess
import cv2 as cv
import numpy as np

img = cv.imread("monitor_1_hr.jpeg")
cv.imshow("Noise", img)

blank = np.zeros(img.shape, dtype = 'uint8')

threshold, n_image = cv.threshold(img, 245, 255, cv.THRESH_BINARY)
cv.imshow("Threshold", n_image)

b,g,r = cv.split(n_image)
cv.imshow("Green Image", g)

bilateral = cv.bilateralFilter(g, 20, 5, 5)
cv.imshow("Bilateral", bilateral)

canny = cv.Canny(bilateral, 200, 255)
cv.imshow("Canny", canny)

contours, higherarcies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
cv.imshow("drawn", blank)

my_config = r"--psm 9 --oem 3"

height, width, chaneels = img.shape

text = tess.image_to_string(canny, config= my_config)
print(text)

cv.waitKey(0)