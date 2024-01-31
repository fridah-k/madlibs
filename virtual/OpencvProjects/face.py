import cv2
import numpy as np
import os
# this line loads the classifier that is the cascade classifier
# face_cascade= cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# cap= cv2.VideoCapture(0)
# cap.set(3,640)# set width
# cap.set(4,480)# set height

# while(True):
#     ret, img = cap.read()
#     gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #gray is the input grayscale image.
# #scale factor is the parameter specifying how much the image size is reduced at each image scale
# #minNeighbors it specifies how many neighbors each candidate rectangle should have
# #minSize is the minimum rectangle size to be considered a face.
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)
# # We mark the faces using the blue triangle.whereby w and h is the height and width
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img,(x,y), (x+w, y+h), (255,0,0),2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = img[y:y+h, x:x+w]
#     cv2.imshow("Face detector - to quit press ESC", img)
#     k= cv2.waitKey(30) & 0xff
#     if k == 27: # press esc to quit
#         break

# cap.release()
# cv2.destroyAllWindows()
