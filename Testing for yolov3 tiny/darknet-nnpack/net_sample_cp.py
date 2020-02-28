# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

import time
import math
from dronekit import connect, VehicleMode, LocationGlobalRelative, Command, LocationGlobal
from pymavlink import mavutil
#from opencv.lib_aruco_pose import *



# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()
        
# Load names of classes
classesFile = "coco.names"
#classesFile = "banana.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
#modelConfiguration = "yolov3.cfg"
#modelWeights = "yolov3.weights"

modelConfiguration = "yolov3-tiny.cfg"
#modelWeights = "yolov3-tiny_last.weights"


#modelConfiguration = "yolov3-tiny.cfg"
modelWeights = "yolov3-tiny.weights"
#modelWeights = "banana.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

def camera_to_uav(x_cam, y_cam):
    x_uav =-y_cam
    y_uav = x_cam
    return(x_uav, y_uav)

def uav_to_ne(x_uav, y_uav, yaw_rad):
    c       = math.cos(yaw_rad)
    s       = math.sin(yaw_rad)
    
    north   = x_uav*c - y_uav*s
    east    = x_uav*s + y_uav*c 
    return(north, east)

def marker_position_to_angle(x, y, z):

    angle_x = math.atan2(x,z)
    angle_y = math.atan2(y,z)

    return (angle_x, angle_y)

def get_location_metres(original_location, dNorth, dEast):
    """
    Returns a Location object containing the latitude/longitude `dNorth` and `dEast` metres from the
    specified `original_location`. The returned Location has the same `alt and `is_relative` values
    as `original_location`.
    The function is useful when you want to move the vehicle around specifying locations relative to
    the current vehicle position.
    The algorithm is relatively accurate over small distances (10m within 1km) except close to the poles.
    For more information see:
    http://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters
    """
    earth_radius=6378137.0 #Radius of "spherical" earth
    #Coordinate offsets in radians
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*math.cos(math.pi*original_location.lat/180))

    print ("dlat, dlon", dLat, dLon)

    #New position in decimal degrees
    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    return(newlat, newlon)

def check_angle_descend(angle_x, angle_y, angle_desc):
    return(math.sqrt(angle_x**2 + angle_y**2) <= angle_desc)

def locate(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    #print("A")
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
   # print("B")
    for out in outs:
        #print("C")
        for detection in out:
            #print("D")
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            #print(classId)
            #print(confidence)
            #print(confThreshold)
	    if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                #boxes.append([left, top, width, height])
                boxes.append([center_x,center_y])
    return len(boxes)>0,boxes,classIds
# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
#if (not args.image):
#    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

time_0 = time.time()
freq_send       = 1 #- Hz

rad_2_deg   = 180.0/math.pi
deg_2_rad   = 1.0/rad_2_deg

land_alt_cm         = 50.0
angle_descend       = 20*deg_2_rad
land_speed_cms      = 30.0

#--------------------------------------------------
#-------------- CONNECTION
#--------------------------------------------------
#-- Connect to the vehicle
#print('Connecting...')
#vehicle = connect(args.connect)

while cv.waitKey(1) < 0:

  # get frame from the video
    hasFrame, frame = cap.read()
    
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        # Release device
        cap.release()
        break
    #print "A"
    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    #print "B"
    # Sets the input to the network
    net.setInput(blob)
    #print "C"
    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))
    #print "D"
    # Remove the bounding boxes with low confidence
    #postprocess(frame, outs)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    # t, _ = net.getPerfProfile()
    # label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    # cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes
    # if (args.image):
    #     cv.imwrite(outputFile, frame.astype(np.uint8))
    # else:
    #     vid_writer.write(frame.astype(np.uint8))
    #print "E"
    found, boxes, class_ids=locate(frame, outs)
    #boxes.append([center_x,center_y])
    if found ==True:
        print(boxes)
        print(boxes[0][0])
	for cid in class_ids:
		print(classes[cid])
"""
        # in boxes we have several coordinates
        # we can move to each one by order
        for box in boxes:
            x_cm=box[0]
            y_cm=box[1]

            x_cm, y_cm          = camera_to_uav(x_cm, y_cm)
  	    print("x is ",x_cm)
	    print("y is ",y_cm)

            uav_location        = vehicle.location.global_relative_frame
            z_cm = uav_location.alt*100.0

            angle_x, angle_y    = marker_position_to_angle(x_cm, y_cm, z_cm)

            if time.time() >= time_0 + 1.0/freq_send:
                time_0 = time.time()
                # print ""
                print(" ")
                print("Altitude = %.0fcm",z_cm)
                print("Marker found x = %5.0f cm  y = %5.0f cm -> angle_x = %5f  angle_y = %5f",x_cm, y_cm,
                      angle_x*rad_2_deg, angle_y*rad_2_deg)

                north, east             = uav_to_ne(x_cm, y_cm, vehicle.attitude.yaw)
                print("Marker N = %5.0f cm   E = %5.0f cm   Yaw = %.0f deg",north, east, vehicle.attitude.yaw*rad_2_deg)

                marker_lat, marker_lon  = get_location_metres(uav_location, north*0.01, east*0.01)
                #-- If angle is good, descend
                if check_angle_descend(angle_x, angle_y, angle_descend):
                    print("Low error: descending")
                    location_marker         = LocationGlobalRelative(marker_lat, marker_lon, uav_location.alt-
                                                                     (land_speed_cms*0.01/freq_send))
                else:
                    location_marker         = LocationGlobalRelative(marker_lat, marker_lon, uav_location.alt)

                vehicle.simple_goto(location_marker)
                print("UAV Location    Lat = %.7f  Lon = %.7f",uav_location.lat, uav_location.lon)
                print("Commanding to   Lat = %.7f  Lon = %.7f",location_marker.lat, location_marker.lon)

                #--- COmmand to land
                if z_cm <= land_alt_cm:
                    if vehicle.mode == "GUIDED":
                        print(" -->>COMMANDING TO LAND<<")
                        vehicle.mode = "LAND"




    #cv.imshow(winName, frame)
"""
