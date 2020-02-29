from picamera import PiCamera
from subprocess import Popen, PIPE
import threading
from time import sleep
import os, fcntl
import cv2
import select
import shutil
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


#--------------------------------------------------
#-------------- CONNECTION
#--------------------------------------------------
#-- Connect to the vehicle
print('Connecting...')
time_0 = time.time()
freq_send       = 1 #- Hz

rad_2_deg   = 180.0/math.pi
deg_2_rad   = 1.0/rad_2_deg

land_alt_cm         = 50.0
hover_alt_cm        =100.0
angle_descend       = 20*deg_2_rad
land_speed_cms      = 30.0
baud_rate = 57600
#connection_string = "/dev/ttyAMA0"
#vehicle = connect(connection_string,baud=baud_rate, wait_ready=True)

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


iframe = 0

camera = PiCamera()

#Yolo v3 is a full convolutional model. It does not care the size of input image, as long as h and w are multiplication of 32

#camera.resolution = (160,160)
#camera.resolution = (416, 416)
#camera.resolution = (544, 544)
camera.resolution = (608, 608)
#camera.resolution = (608, 288)


camera.capture('frame.jpg')
sleep(0.1)

#spawn darknet process

yolo_proc = Popen(["./darknet",
                   "detect",
                   "./cfg/yolov3-tiny0802.cfg",
                   "./yolov3-tiny0802_4000.weights",
                   "-thresh","0.1"],
                   stdin = PIPE, stdout = PIPE)


fcntl.fcntl(yolo_proc.stdout.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)

stdout_buf = ""
while cv2.waitKey(1) < 0:
    try:
        select.select([yolo_proc.stdout], [], [])
        stdout = yolo_proc.stdout.read()
        stdout_buf += stdout
        if 'Enter Image Path' in stdout_buf:
            try:
               im = cv2.imread('predictions.png')
               #print(im.shape)
               cv2.imshow('yolov3-tiny',im)
               #key = cv2.waitKey(5) 
            except Exception as e:
                print("Error:", e)
            camera.capture('frame.jpg')
            yolo_proc.stdin.write('frame.jpg\n')
            stdout_buf = ""
        if len(stdout.strip())>0:
            print('get %s' % stdout)
              
        with open("results.txt","r") as file:
            for line in file:
                fields=line.split(";")
                x_cm=float(fields[0])
                y_cm=float(fields[1])
                x_cm, y_cm          = camera_to_uav(x_cm, y_cm)
                print("x is ",x_cm)
                print("y is ",y_cm)
        
                uav_location        = vehicle.location.global_relative_frame
                z_cm = uav_location.alt*100.0

                angle_x, angle_y    = marker_position_to_angle(x_cm, y_cm, z_cm)

                if time.time() >= time_0 + 1.0/freq_send:
                    time_0 = time.time()
                    print(" ")
                    print("Altitude = %.0fcm",z_cm)
                    print("Marker found x = %5.0f cm  y = %5.0f cm -> angle_x = %5f  angle_y = %5f",x_cm, y_cm,
                    angle_x*rad_2_deg, angle_y*rad_2_deg)

                    north, east             = uav_to_ne(x_cm, y_cm, vehicle.attitude.yaw)
                    print("Marker N = %5.0f cm   E = %5.0f cm   Yaw = %.0f deg",north, east, vehicle.attitude.yaw*rad_2_deg)

                    marker_lat, marker_lon  = get_location_metres(uav_location, north*0.01, east*0.01)
                  # angle checking was removed as altitude doesn’t change anymore
                    location_marker         = LocationGlobalRelative(marker_lat, marker_lon, hover_alt_cm)

                    vehicle.simple_goto(location_marker)
                    print("UAV Location    Lat = %.7f  Lon = %.7f",uav_location.lat, uav_location.lon)
                    print("Commanding to   Lat = %.7f  Lon = %.7f",location_marker.lat, location_marker.lon)

          
    except Exception as e:
        print("Error:", e)
