#Program to Detect the Face and Recognise the Person based on the data from face-trainner.yml
import cv2
import numpy as np
import os
from PIL import Image

labels= ["Karimi", "Robert"]

face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
#load the face-trainner.yml file into our program since we will have to use the data from that file to recognize faces.
recognizer.read("face-trainner.yml")

cap = cv2.VideoCapture(0) #Get video feed from the Camera
while (True):
    ret, img= cap.read() # break video into frames
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)# convert the video into grayscale
    faces= face_cascade.detectMultiScale(gray)# recognize the faces
    
    for (x, y, w, h) in faces:
        roi_gray= gray[y:y+h, x:x+w]# convert face to grayscale

        id_, conf= recognizer.predict(roi_gray)# recognize the face
        print('--------------------------------------')
        print(roi_gray)
        print('--------------------------------------')
        
#The variable conf tells us how confident the software is recognizing the face.
#if the confidence level is greater than 80, 
# we get the name of the person using the ID number.    

        
        if conf>=80:
            
            font=cv2.FONT_HERSHEY_SIMPLEX # font style for the name
            if(id_ <  len(labels)):
                name = labels[id_]
            else:
                name = 'Not Found'
            
            cv2.putText(img, name, (x,y), font, 1, (0,0,255), 2)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('preview', img) # display the video
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
# release the capture
cap.release()
cv2.destroyAllWindows()