#-------------------------------------------
# SEGMENT HAND REGION FROM A VIDEO SEQUENCE AND WRITE THEM INTO FOLDER TO MAKE DATA TO TRAIN
#-------------------------------------------

#--------------------------------------------------
# First we use the background subtraction algorithm, threshold and coutour to detect hand and blackout the
# unnecessary parts. We write these threshold images into a file which is used to train the model
#--------------------------------------------------

# organize imports
 
from PIL import Image
import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise

# global variables
bg = None #background

#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    
    # get the contours in the thresholded image
    cnts, tmp = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#-----------------
# Resize the images so that it can be fed into the Convolution Neural Network which accepts 89 x 100 dimensional image.
#-----------------
def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":
    # initialize accumulated weight
    accumWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames and num of image in dataset file
    num_frames = 0
    num_image = 0

    start_record = False

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our weighted average model gets calibrated
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successfull...")
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                if (num_image < 300 and start_record) :
                    #Mention the directory in which file you wanna store the image followed by the image name
                    cv2.imwrite("Dataset/Swing/swing_" + str(num_image) + ".png", thresholded)
                    num_image += 1

                # show the thresholded image
                cv2.imshow("Thesholded", thresholded)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

        # if the user pressed "s", then start recording
        if keypress == ord("s"):
            start_record = True
    
    # Resize all the image after write them in their file
    for i in range(1, num_image):
    # Mention the directory in which you wanna resize the images followed by the image name
        resizeImage("Dataset/Swing/swing_" + str(i) + ".png")

# free up memory
camera.release()
cv2.destroyAllWindows()