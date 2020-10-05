import numpy as np
import cv2 as cv
import heapq
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation =cv.INTER_AREA)

def draw_centers_of_largest_contours(img, n, contours, color, size, offset=(0,0)):
    largest_n_contours = heapq.nlargest(n,contours,key=cv.contourArea)
    cv.drawContours(img,largest_n_contours,-1,color,size, offset=offset)
    for c in largest_n_contours:
        M = cv.moments(c)
        if M["m00"] != 0.0:
        # Offset due to the processing on an extracted image (x,y) + (center in other imagex, center in other image y)
            cX = offset[0] + int(M["m10"] / M["m00"])
            cY = offset[1] + int(M["m01"] / M["m00"])
            cv.circle(img, (cX, cY), size, color, -1)

def draw_color_histogram(img, colors=['r','g','b'], color_labels=['Red','Green','Blue']):
    assert len(colors) == len(color_labels), 'Each color does not have a corresponding color_label.'
    plt.figure()
    legend_keys = []
    for i in range(len(colors)):
        legend_keys.append(mpatches.Patch(color=colors[i], label=color_labels[i]))
    plt.legend(handles=legend_keys)
    for i,col in enumerate(colors):
        histr = cv.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()

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
            
            # Dilation to connect the wheel more
            dilated = cv.dilate(threshold, cv.getStructuringElement(cv.MORPH_ELLIPSE, (20,20)))
            # cv.imshow("dilated", dilated)

             # Getting Contours
             # Remove Dependence on Hardcoded Values
             # Plot Histograms Across the Duration of the Video (and see how they fluctuate)
            contours, _ = cv.findContours(dilated,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                largest_countour = max(contours,key=cv.contourArea)
                x,y,w,h = cv.boundingRect(largest_countour)
                spinner = img_in[y:y+h, x:x+w]
                cv.imshow("spinner",spinner)

                draw_color_histogram(spinner,colors=['b','g','r'],color_labels=['Blue','Green','Red'])
                draw_color_histogram(cv.cvtColor(spinner,cv.COLOR_BGR2HSV),colors=['c','y','m'],color_labels=['Hue','Saturation','Value'])

                # Blur on Spinner
                blurred = cv.blur(spinner, (15,15))
                # cv.imshow("blurred",blurred)

                hsv = cv.cvtColor(spinner, cv.COLOR_BGR2HSV)
                # 2nd layer of thresholding (For Red Spinner Part)
                hsv_threshold_red = cv.inRange(hsv,(0,0,0),(179,70,255))
                # cv.imshow("hsv threshold for red spinner", hsv_threshold_red)

                # Getting Center of HSV Contours for Red
                hsv_contours, _ = cv.findContours(hsv_threshold_red,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
                draw_centers_of_largest_contours(img_in, 2, hsv_contours, (0,0,255), 3, offset=(x,y))

                # 2nd layer of thresholding (For Black Spinner Part)
                hsv_threshold_black = cv.inRange(hsv,(0,190,0),(179,255,160))
                # cv.imshow("hsv threshold for black spinner", hsv_threshold_black)

                # Getting Center of HSV Contours for Black
                hsv_contours, _ = cv.findContours(hsv_threshold_black,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
                draw_centers_of_largest_contours(img_in, 2, hsv_contours, (0,0,0), 3, offset=(x,y))

                # 2nd layer of thresholding (For Green Spinner Part)
                # Equalize Histogram for 3 color channels
                # Plot histograms before and after
                grey_spinner = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
                equalized_spinner = cv.equalizeHist(grey_spinner)
                # cv.imshow('equalized_spinner', equalized_spinner)

                # Green Detection Algorithm
                roulette_rim = cv.inRange(equalized_spinner, 190, 255)
                # cv.imshow('roulette_rim',roulette_rim)
                hsv_threshold_green = cv.inRange(hsv,(80,212,0),(179,255,230))

                # green_binarized_image = hsv_threshold_green & !roulette_rim & !hsv_threshold_black & !hsv_threshold_red
                # add confidence score (heuristics)
                green_binarized_image = cv.bitwise_and(hsv_threshold_green,cv.bitwise_and(cv.bitwise_not(roulette_rim),cv.bitwise_and(cv.bitwise_not(hsv_threshold_black),cv.bitwise_not(hsv_threshold_red))))
                green_contours, _ = cv.findContours(green_binarized_image,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
                draw_centers_of_largest_contours(img_in, 2, green_contours, (0,255,0), 3, offset=(x,y))

                # Showing Spinner Bounding Box
                cv.rectangle(img_in,(x,y),(x+w,y+h),(0,255,0),2)

            # cv.drawContours(img_in,contours,-1,(100,0,225),3)
            # cv.imshow("saliency", saliencyMap)
            # cv.imshow("threshold", threshold)

        cv.imshow('img_in',img_in)



    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()