import pytesseract as tess
import numpy as np
import cv2 as cv

img = cv.imread("monitor_1_rr.jpeg")
cv.imshow("rr", img)

blank = np.zeros(img.shape, dtype= 'uint8')
gray_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

noise_reduced = cv.fastNlMeansDenoising(gray_scale, None, 10, 3, 3)
cv.imshow("Noise reduced", noise_reduced)

thresh, n_image = cv.threshold(noise_reduced, 200, 255, cv.THRESH_BINARY)
cv.imshow("Thresholded adjusted image", n_image)

bilateral = cv.bilateralFilter(n_image, 30, 20, 20)
cv.imshow("Bilateral", bilateral)

canny = cv.Canny(bilateral, 225, 255)
cv.imshow("Canny", canny)

contours, higherarcies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
cv.imshow("drawn", blank)

my_config = r"--psm 8 --oem 3"

height, width, chaneels = img.shape

text = tess.image_to_string(n_image, config= my_config)
print(text)

cv.waitKey(0)