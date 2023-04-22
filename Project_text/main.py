import pytesseract as tess
import PIL.Image as pil
import cv2 as cv

my_config = r"--psm 6 --oem 3"

text = tess.image_to_string(pil.open("image.png"), config= my_config)
print(text)