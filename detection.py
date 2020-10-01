import numpy as np
import cv2 as cv

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation =cv.INTER_AREA)

cap = cv.VideoCapture('/Users/karthikdharmarajan/Documents/URobotics/Course Footage/GOPR1145.MP4')
saliency = cv.saliency.StaticSaliencySpectralResidual_create()

while cap.isOpened():
    ret, img_in = cap.read()
    
    if ret:
        img_in = rescale_frame(img_in,30)

        # Saliency detects the foreground
        (success, saliencyMap) = saliency.computeSaliency(img_in)
        if success:
            saliencyMap = (saliencyMap * 255).astype("uint8")

            # Thresholding
            ret2,threshold = cv.threshold(saliencyMap,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
            
            dilated = cv.dilate(threshold, cv.getStructuringElement(cv.MORPH_ELLIPSE, (30,30)))
            cv.imshow("dilated", dilated)

             # Getting Contours
            contours, _ = cv.findContours(dilated,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(img_in,contours,-1,(100,0,225),3)
            if len(contours) > 0:
                largest_countour = max(contours,key=cv.contourArea)
                x,y,w,h = cv.boundingRect(largest_countour)
                cv.rectangle(img_in,(x,y),(x+w,y+h),(0,255,0),2)

            cv.imshow("saliency", saliencyMap)
            cv.imshow("threshold", threshold)

        cv.imshow('img_in',img_in)



    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()