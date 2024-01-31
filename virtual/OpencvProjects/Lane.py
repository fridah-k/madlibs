# A project on road lane line detection. detecting the white markings on the road.
import numpy as np
import cv2
import matplotlib.pyplot as plt
# load the image
image = cv2.imread('/home/fridah/Downloads/solidWhiteRight.jpg')
# convert the image into the RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# lets get the shape of the image
print(image.shape)
height = image.shape[0]
width = image.shape[1]
# lets define our region of interest
region_of_interest_vertices = [(0, height), (width/2, height/2), (width, height)]
# a function to mask everything but not the region of interest.
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = (255,)
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# detecting the lane lines
# convert the image into gray image
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
canny_image = cv2.Canny(gray_image, 100, 200)
# to get rid of the unwanted edges
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
canny_image = cv2.Canny(gray_image, 100, 200)
masked_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # if there are no lines to draw exit. and if there is return none
    # make a copy of the original image
    img = np.copy(image)
    # create a 
    line_img = np.zeros((image.shape[0], image.shape[1]))
    for x1, y1, x2, y2 in line:
        cv2.line(image, (x1, y1), (x2, y2), (255,0,0), thickness=3)
        # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img

# to draw lines using the Hough transformation

lines = cv2.HoughLinesP(masked_image,rho=6, theta= np.pi/60, threshold=160, 
lines=np.array([]), minLineLength=40, maxLineGap=25)
# we can iterate over the output lines and draw them
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(image, (x1, y1), (x2, y2), (255,0,0), thickness=3)

# Display the image
plt.imshow(image)
plt.imshow(masked_image)
plt.imshow(canny_image)
plt.imshow(image)
plt.show()







