import numpy as np
import cv2 
img = cv2.imread("/home/fridah/Desktop/German-shepherd.jpg", cv2.IMREAD_COLOR)
# converting the image into gray scale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# convert the image to blur 
imgBlur = cv2.GaussianBlur(imgGray, (7,7), 0)
# to get the edge detectors of the image
imgCanny = cv2.Canny(img, 100,100)
# to increase the thickness of our edges we use the dialate function.
kernel = np.ones((5,5), np.uint8)
imgDialation = cv2.dilate(imgCanny, kernel, iterations = 1)
# the opposite of dialation is eroded
imgEroded = cv2.erode(imgDialation, kernel, iterations = 1)
# to cartooninfy the image
edges = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
cv2.THRESH_BINARY, 9, 9)
color = cv2.bilateralFilter(img, 9, 250, 250)
cartoon = cv2.bitwise_and(color, color, mask=edges)
# joining of the images
imgHor = np.hstack((img, img))
imgVer = np.vstack((img, img))
# resizing of the images
scale_percent = 40 # percent of the original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
imgResize = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
print('Resized Dimensions : ',imgResize.shape)
# cropping of images
imgCropped = img[0:200, 200:500]
cv2.imshow("Image Cropped", imgCropped)
cv2.imshow("Image", img)
cv2.imshow("ImageResize", imgResize)
cv2.imshow("Horizontal", imgHor)
cv2.imshow("Vertical", imgVer)
cv2.imshow("edges", edges)
cv2.imshow("Cartoon", cartoon)
cv2.imshow("ErodedImage", imgEroded)
cv2.imshow("DilateImage", imgDialation)
cv2.imshow("CannyImage", imgCanny)
cv2.imshow("BlurImage", imgBlur)
cv2.imshow("GrayImage", imgGray)
cv2.imshow("img", img)
cv2.waitKey(0)
