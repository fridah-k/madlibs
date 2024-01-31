import numpy as np #For converting Images to Numerical array
import cv2 #For Image processing
import os #To handle directories
from PIL import Image #Pillow lib for handling images

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# recognizer = cv2.createLBPHFaceRecognizer()
recognizer = cv2.face.LBPHFaceRecognizer_create()

Face_ID = -1 
pev_person_name = ""
y_ID = []
x_train = []

# tell the progarm where you have saved the image
Face_images= os.path.join(os.getcwd(), "Face_images")
print(Face_images)

for root, dirs, files in os.walk(Face_images):
    for file in files: #check every directory in it 
        if file.endswith("jpeg") or file .endswith("jpg") or file.endswith("png"):
            path= os.path.join(root, file)
            person_name = os.path.basename(root)
            print(path, person_name)
            if pev_person_name!=person_name:#Check if the name of person has changed
                Face_ID=Face_ID+1#If yes increment the ID count 
                pev_person_name= person_name
            Gray_Image= Image.open(path).convert("L") # convert the image to gray scale using pillow
#Crop the Grey Image to 550*550 (Make sure your face is in the center in all image)
            Crop_Image= Gray_Image.resize((550, 550), Image.ANTIALIAS)
            Final_Image = np.array(Crop_Image)
            faces= face_cascade.detectMultiScale(Final_Image)
            print (Face_ID, faces)

            for (x,y,w,h) in faces:
                roi= Final_Image[y:y+h, x:x+w] #crop the Region of Interest (ROI)
                x_train.append(roi)
                y_ID.append(Face_ID)
recognizer.train(x_train, np.array(y_ID))#Create a Matrix of Training data 
recognizer.save("face-trainner.yml") #Save the matrix as YML file 


            

            
                