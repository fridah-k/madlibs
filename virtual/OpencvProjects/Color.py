# Detecting color of the images
import numpy as np
import pandas as pd
import cv2
import argparse
# load the image
img_path = "/home/fridah/Downloads/colorpic.jpg"
img = cv2.imread(img_path)

# Read the csv file with pandas and giving names to each column
index = ["color", "color_name", "hex", "R", "G", "B"]
# hex is the value of the color
csv = pd.read_csv("/home/fridah/Downloads/colors.csv", names=index, header=None)

#Declaring Global variables
clicked = False
r = g = b = xpos = ypos = 0
# xposition and y position they are all declared as zero.

# Function to calculate minimum distance from all colors and get the most matching color
def get_color_name(R,G,B):
    minimum = 10000
    for i in range(len(csv)): # it means we are looping through all the columns
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"])) + abs(B- int(csv.loc[i,"B"]))
        # d stands for the distance in the values of all columns
        # abs  stands for absolute that means the values in the difference of the colors will be absolute.
        # the absolute value means that it will be positive even if the value turns out to be negative

        if (d<=minimum): # if d which is the total of all values
            minimum = d
            cname = csv.loc[i,"color_name"]
    return cname

# function to get the x, y coordinates of mouse double click
def mouse_click(event, x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, xpos,ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b,g,r = img[y,x]
        b = int(b)
        g = int(g)
        r = int(r)
cv2.namedWindow('color detection')
cv2.setMouseCallback('color detection',mouse_click)

while(1):
    cv2.imshow("color detection",img)
    if clicked:
        #cv2.rectangle(image, startpoint, end point,color, thickness)-1 fills entire rectangle
        cv2.rectangle(img,(20,20), (750,60), (b,g,r), -1)
        # creating a text string to display (color name and RGB values)
        print('r'+ str(r)+', g'+ str(g)+', b'+str(b))
        text = get_color_name(r,g,b) + 'R='+ str(r) + 'G='+ str(g) + 'B='+ str(b)
        #cv2.putText(img, text, start, font(0-7), fontscale,thickness ,line type)
        cv2.putText(img, text,(50,50),2,0.8,(255,0,0),2,cv2.LINE_AA)
        # for very light colors we will display the text in black colours
        if (r+g+b>=600):
            cv2.putText(img, text,(50,50),2,0.8,(0,0,0),2,cv2.LINE_AA)
        clicked=False
# break the loop when the user hits esc key
    if cv2.waitKey(20) & 0xFF ==27:
        break
cv2.destroyAllWindows()

