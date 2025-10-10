import cv2 as cv

img = cv.imread('dataset/Laptop/train/images/laptop.jpg')

cv.imshow('image', img)

cv.waitKey(0)
cv.destroyAllWindows()

