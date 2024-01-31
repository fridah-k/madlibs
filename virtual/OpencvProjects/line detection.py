import numpy as np
import cv2
# we want to confirm if the image or the video is running
def canny(img):
    if img is None:
       cap.release()
       cv2.destroyAllWindows()
       exit()
    kernel = 5
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (kernel,kernel), 0)
    canny = cv2.Canny(gray, 50,150)
    return canny
# Define our region of interest in our canny image
def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)
    triangle = np.array([[(200, height),(800, 350), (1200, height),]], np.int32)
    # this are the three points of the triangle that it takes 200, height and the rest
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

def houghLines(cropped_canny):
    return cv2.HoughLinesP(cropped_canny, rho=2, theta=np.pi/180, threshold=100, 
        np.array([[(200, height),(800, 350), (1200, height),]], minLineLength=40, maxLineGap=5)

    # adding weight to the lines so that it can be visible 
    # we are adding weight on the frame and on the image
                                             def addWeighted(frame,line_image): 
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    
# we are drawing lines on the image.(0,0,255 its the color and 10 is the thickness)
def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x, y2 in line:
                cv2.line(line_image, (x1,y1), (x2,y2), (0,0,255),10) # we are adding the red color on the orinal image for visibility
    return line_image

# defining the slope of the lines
# the image sizes are defined by y1 y2 x1 and x2
# the line should be by the following slope (y1*3.0/5)
def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])
    y2 = int(y1*3.0/5)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]

    
# we are defining the average slope and lines for the road
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_points(image, left_fit_average)
    right_fit = make_points(image, left_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines

# read the video file
cap = cv2.VideoCapture('/home/fridah/Downloads/solidWhiteRight.mp4')
#Read the frames of the video
while(cap.isOpened()):
    a, frame = cap.read()# read the image that is being displayed by the cap()
    #the first image manipulation is:
    canny_image = canny(frame)
    cropped_canny = region_of_interest(canny_image)
    cv2.imshow("CannyImage", canny_image)
    #cv2.imshow("Cropped_canny", cropped_canny)
# Hough lines and average are used to detect lines in the lanes
    lines = houghLines(cropped_canny)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = addWeighted(frame, line_image)
    cv2.imshow("result", combo_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

