import numpy as np
import argparse
import cv2 as cv
import heapq
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

created_histogram = False
figure = None
axis = None
individual_histograms = []

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation =cv.INTER_AREA)

def draw_centers_of_largest_contours(img, n, contours, color, size, offset=(0,0), f=cv.contourArea):
    largest_n_contours = heapq.nlargest(n,contours,key=f)
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
    global figure, axis, created_histogram, individual_histograms
    if not created_histogram:
        figure, axis = plt.subplots()
        legend_keys = []
        for i in range(len(colors)):
            legend_keys.append(mpatches.Patch(color=colors[i], label=color_labels[i]))
            component_data, = axis.plot(np.arange(256), np.zeros((256,)), c=colors[i], label=color_labels[i])
            individual_histograms.append(component_data)
        plt.legend(handles=legend_keys)
        axis.set_xlim([0,256])
        axis.set_ylim([0,5000])
        plt.show(block=False)
        created_histogram = True
    else :
        for i in range(len(colors)):
            histr = cv.calcHist([img],[i],None,[256],[0,256])
            individual_histograms[i].set_ydata(histr)
            figure.canvas.draw()

def heuristic(contour):
    rect = cv.minAreaRect(contour)
    area = rect[1][0] * rect[1][1]
    diff = cv.contourArea(cv.convexHull(contour)) - cv.contourArea(contour)
    cent = rect[0]
    heur = 3 * area - 9 * diff
    return heur

parser = argparse.ArgumentParser(description='Roulette Wheel Vision')
parser.add_argument('-hist', action='store_true', help='Outputs a histogram every n frames specified by -n', default=False)
parser.add_argument('-n', help='The amount of frames after which a histogram is displayed', type=int, default=100)
parser.add_argument('-equalized', action='store_true', help='This flag sets if the histogram displayed is after image histogram equalization', default=False)
args = parser.parse_args()
cap = cv.VideoCapture('/Users/karthikdharmarajan/Documents/URobotics/Course Footage/GOPR1145.MP4')
saliency = cv.saliency.StaticSaliencySpectralResidual_create()
count = 0

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
            dilated = cv.dilate(threshold, cv.getStructuringElement(cv.MORPH_ELLIPSE, (30,30)))

            # Getting Contours
            contours, _ = cv.findContours(dilated,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                largest_countour = max(contours,key=cv.contourArea)
                x,y,w,h = cv.boundingRect(largest_countour)
                spinner = img_in[y:y+h, x:x+w]

                hsv_spinner = cv.cvtColor(spinner,cv.COLOR_BGR2HSV)
                blurred_spinner = cv.bilateralFilter(hsv_spinner,7,50,50)
                bgr_blurred_spinner = cv.cvtColor(blurred_spinner,cv.COLOR_HSV2BGR)
                # cv.imshow('blurred_spinner', bgr_blurred_spinner)

                # Index 0: Hue, Index 1: Saturation, Index 2: Value
                individual_channels = cv.split(blurred_spinner)

                individual_channels[0] = cv.equalizeHist(individual_channels[0])
                individual_channels[1] = cv.equalizeHist(individual_channels[1])
                individual_channels[2] = cv.equalizeHist(individual_channels[2])

                equalized_image = cv.merge((individual_channels[0],individual_channels[1],individual_channels[2]))
                # cv.imshow('equalized_image', cv.cvtColor(equalized_image,cv.COLOR_HSV2BGR))

                # Background Green Detection
                Lab_spinner = cv.cvtColor(bgr_blurred_spinner, cv.COLOR_BGR2LAB)
                Lab_individual_channels = cv.split(Lab_spinner)
                _ , L_threshold = cv.threshold(Lab_individual_channels[0],1,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
                _ , a_threshold = cv.threshold(Lab_individual_channels[1],1,255,cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
                b_threshold_value, b_threshold_background = cv.threshold(Lab_individual_channels[2],1,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
                b_bias = 3
                b_threshold_background_value = b_bias + b_threshold_value
                _, b_threshold_background = cv.threshold(Lab_individual_channels[2],b_threshold_background_value,255,cv.THRESH_BINARY)
                cv.imshow('b_threshold_background',b_threshold_background)


                b_edges_bias = 5
                b_threshold_edges = b_threshold_value - b_edges_bias
                _, b_threshold_edges = cv.threshold(Lab_individual_channels[2],b_threshold_edges,255,cv.THRESH_BINARY_INV)
                cv.imshow('b_threshold_edges',b_threshold_edges)

                if args.hist:
                    if count % args.n == 0:
                        draw_color_histogram(equalized_image if args.equalized else blurred_spinner,colors=['c','y','m'],color_labels=['Hue','Saturation','Value'])
                    count += 1

                # General Thresholds
                _, hue_threshold_norm = cv.threshold(individual_channels[0],0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
                _, saturation_threshold_inv = cv.threshold(individual_channels[1],0,255,cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
                _, value_threshold_inv = cv.threshold(individual_channels[2],0,255,cv.THRESH_BINARY_INV|cv.THRESH_OTSU)

                # red_threshold = hue_threshold_norm & saturation_threshold_inv & value_threshold_inv
                red_threshold = cv.bitwise_and(hue_threshold_norm, cv.bitwise_and(saturation_threshold_inv, value_threshold_inv))
                red_contours, _ = cv.findContours(red_threshold,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
                draw_centers_of_largest_contours(img_in,2,red_contours,(0,0,255),3,offset=(x,y))

                # black_threshold = hue_threshold_norm & !saturation_threshold_inv & value_threshold_inv
                black_threshold = cv.bitwise_and(hue_threshold_norm, cv.bitwise_and(cv.bitwise_not(saturation_threshold_inv), value_threshold_inv))
                black_contours, _ = cv.findContours(black_threshold,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
                draw_centers_of_largest_contours(img_in,2,black_contours,(0,0,0),3,offset=(x,y))

                # green_threshold = !hue_threshold_norm & !saturation_threshold_inv & !value_threshold_inv & !b_threshold_background & !b_threshold_edges
                green_threshold = cv.bitwise_and(cv.bitwise_and(
                    cv.bitwise_and(cv.bitwise_not(hue_threshold_norm), cv.bitwise_and(cv.bitwise_not(saturation_threshold_inv), cv.bitwise_not(value_threshold_inv)))
                    ,cv.bitwise_not(b_threshold_background)),cv.bitwise_not(b_threshold_edges))
                kernel = np.ones((4,4),np.uint8)
                green_threshold = cv.morphologyEx(green_threshold,cv.MORPH_OPEN,kernel)
                cv.imshow("green_threshold", green_threshold)
                green_contours, _ = cv.findContours(green_threshold,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
                draw_centers_of_largest_contours(img_in,2,green_contours,(0,255,0),3,offset=(x,y), f=heuristic)

                # Showing Spinner Bounding Box
                cv.rectangle(img_in,(x,y),(x+w,y+h),(0,255,0),2)

            # cv.imshow("threshold", threshold)

        cv.imshow('img_in',img_in)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()