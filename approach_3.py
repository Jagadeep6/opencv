import pytesseract as tess
import numpy as np
import cv2 as cv

img = cv.imread("monitor_3.jpeg")
cv.imshow("noise", img)
blank = np.zeros(img.shape, dtype= 'uint8')
noise_reduced = cv.fastNlMeansDenoisingColored(img, None, 5, 10, 7, 21)

thresh, t_img = cv.threshold(noise_reduced, 235, 255, cv.THRESH_BINARY_INV)
t_img = cv.cvtColor(t_img, cv.COLOR_BGR2GRAY)
cv.imshow("thresh", t_img)

canny = cv.Canny(t_img, 225, 255)
cv.imshow("Canny", canny)

contours, higherarcies = cv.findContours(t_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
cv.imshow("drawn", blank)


cv.imshow("Noiseless",noise_reduced)


my_config = r"--psm 11 --oem 3"
#6, 7 is showing something
#8 is a bit more accurate for bp


height, width, chaneels = img.shape

text = tess.image_to_string(t_img, config= my_config)
print(text)

cv.waitKey(0)