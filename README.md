# Object Detection and drone landing

By Ilgar Rasulov (Ilgar.Rasulov@hsrw.org), Md. Rakib Hassan (Md-Rakib.Hassan@hsrw.org) and Prabhat Kumar (Prabhat.Kumar@hsrw.org).
### (neural network for object detection) - Darknet can be used on [Linux](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux) and [Windows](https://github.com/AlexeyAB/darknet#how-to-compile-on-windows-using-vcpkg)

More details: http://pjreddie.com/darknet/yolo/

### Table of contents
0.  [Introduction](#introduction)
1.  [Raspberry Pi setup](#raspberry-pi-setup)
    * [Remote Access to the Raspberry Pi](#remote-access-to-the-raspberry-pi)
    * [Camera Calibration and code explanation](#camera-calibration-and-code-explanation)
2.  [Drone part](#drone-part)
3.  [Raspberry Pi to Pixhawk connection](#raspberry-pi-to-pixhawk-connection)
    * [Hardware Communication](#hardware-communication)
    * [Configuring serial port](#configuring-serial-port)
    * [Testing connection](#testing-connection)
4.  [Landing with aruco marker approach](#landing-with-aruco-marker-approach)
    * [Aruco marker](#aruco-marker)
    * [Marker creation](#marker-creation)
    * [Marker detection](#marker-detection)
    * [Landing algorithm and code explanation](#landing-algorithm-and-code-explanation)
5.  [Troubleshooting](#troubleshooting)
6.  [Computer Vision](#computer-vision)
    * [How do we build a Deep Learning model](#how-do-we-build-a-deep-learning-model)
7.  [Yolo implementation](#yolo-implementation)
    * [Data Collection](#data-collection)
    * [Data Annotation](#data-annotation)
    * [GPU Support](#gpu-support)
    * [Training tiny yolo v3 in Google Colab](#training-tiny-yolo-v3-in-google-colab)
    * [NNPack](#nnpack)
    * [Testing the model](#testing-the-model)
8.  [Landing algorithm with Object Detection](#landing-algorithm-with-object-detection)
    * [Single object detection and landing](#single-object-detection-and-landing)
    * [Multiple objects detection and landing](#multiple-objects-detection-and-landing)
    * [Hovering over objects](#hovering-over-objects)

### Introduction
On the wave of current trends in computer vision and drone applications, we are introducing you to a project that we worked on “Landing of Drone on an object”. In general, this means making a drone land on any object by using a landing algorithm and a deep learning algorithm for the detection of an object. We choose the state-of-the-art YOLO algorithm as the object detection algorithm. In this project, our final goal was to land a drone on an object. Therefore, in the first phase, we researched and found an existing solution where the drone was landed on an Aruco marker. In the second phase, we trained the YOLO algorithm to recognize an object (Banana) and modified the landing algorithm to land on that object. Then we further modified the landing algorithm to hover over an object.

In this git laboratory, all the procedures and steps needed to reach the goals of this project were written in detail. During the different phases of our project, we faced a lot of challenges. A special thanks to Tiziano Fiorenzani for whom this project was successful. His YouTube videos and tutorials from GitHub helped us a lot.

References: 

Tiziano Fiorenzani - Drone Tutorials

https://www.youtube.com/watch?v=TFDWs_DG2QY&list=PLuteWQUGtU9BcXXr3jCG00uVXFwQJkLRa

AlexeyAB's Darknet repo:

https://github.com/AlexeyAB/darknet/#how-to-train-tiny-yolo-to-detect-your-custom-objects

YOLv3 Series by Ivan Goncharov

https://www.youtube.com/watch?v=R0hipZXJjlI&list=PLZBN9cDu0MSk4IFFnTOIDihvhnHWhAa8W 

Ivangrov's Darknet repo:

https://github.com/ivangrov/YOLOv3-Series

Colab Like a Pro by Ivan Goncharov

https://www.youtube.com/watch?v=yPSphbibqrI&list=PLZBN9cDu0MSnxbfY8Pb1gRtR0rKW4RKBw

DroneKit-Python project documentation:

https://dronekit-python.readthedocs.io/en/latest/


### Raspberry Pi setup
#### Remote Access to the Raspberry Pi

In the testing phase, the communication between the drone and Raspberry Pi was established with Python codes. Normally, Raspberry Pi requires a monitor and a keyboard to work according to your requirements. But these hardwires are not possible to attach to the drone while flying. Therefore, the connection over a network was decided as the optimal solution. We also wanted to run and observe the output of the python scripts on our laptop, so that we can make changes in codes accordingly.
To access the Raspberry Pi over a network, both Raspberry and PC/laptop should be connected on the same internet, and the IP address of Raspberry Pi should be defined. The first approach was to connect both the devices to a common WLAN. Unfortunately, as the university WLAN didn’t give access to connected devices’ IP addresses as well as third-party tools were not reliable enough, this approach became inoperable. Instead, the second approach where both devices were connected to the same WiFi hotspot was preferred.
Windows 10 provides the feature of Mobile Hotspot setting:

![](images/mob_hotspot.jpg)

In this dialog network name (SSID), password and band should be set. For band options of 2.4 and 5 GHz are available. Raspberry Pi didn’t connect to 5 GHz band, so 2.4 GHz was chosen.

Network on Rpi also should be configured. The easy way to do is to use a monitor, keyboard, and mouse only once to connect to a hotspot through the standard menu:

![](images/rpiwifi.png)

https://www.circuitbasics.com/how-to-set-up-wifi-on-the-raspberry-pi-3/ 

If monitor and input devices are not accessible for the initial setup, the configuration files should be added to the SD Card with raspbian image.

Steps:

1) Insert SD card to PC running Linux. If PS works on Windows, Linux can be installed as a guest OS via virtualization tools. The setup steps for VirtualBox software is given below:
  
1.1) open Command Prompt as Administrator (Windows key + “x” → Command prompt (admin))

1.2) list all drives by typing

```c
wmic diskdrive list brief 
```
The output will be similar to that

![](images/mob_hotspot3.jpg)

1.3) insert an SD card and run the command from 1.2 again. The output will be different:

![](images/mob_hotspot4.jpg)

Remember DeviceID, in this case, it is “\\.\PHYSICALDRIVE1”

1.4) navigate to VirtualBox folder

![](images/mob_hotspot5.jpg)

1.5) create Virtual Machine Disk file (VMDK), use as an argument device id (\\.\PHYSICALDRIVE1)

![](images/mob_hotspot6.jpg)

If the file was successfully created, the message will be shown

![](images/mob_hotspot7.jpg)

1.6) Launch VirtualBox as an administrator

![](images/wificonf1.png)

1.7) Navigate to “Settings” > “Storage”.

1.8) Click on “Controller: SATA”.

1.9) Check “Use Host I/O Cache” checkbox.

1.10) click on “Adds hard disk” icon.

![](images/wificonf2.png)

1.11) Select “Choose Existing Disk”.

1.12) Select “C:\sdcard.vmdk” file which created in Step 1.5.

1.13) Launch the virtual machine and verify that the sd card appears in the list of devices. For that run the “fdisk -l” command. In this example, Linux recognized the sd card as /dev/sdb

![](images/RaspberryPi_connection2.png)

VirtualBox steps are over at this point. Further steps are provided for the Linux environment.

1.14) Disk should be mounted to be accessible in Linux

Type the command 

“mkdir /mnt/SD”

to create a mount point. Change /mnt/sD with your path.

1.15) Type the command

"mount /dev/sdb1 /mnt/SD"

to mount the sd card

1.16) Type the command "cd /mnt/SD" to access the files on the SD card.

2) Navigate to boot directory from the mound point
cd boot
3) Add “wpa_supplicant.conf” by typing

nano wpa_supplicant.conf

4) Enter the following text with your wifi name and password

```c
network={
    ssid="YOUR_NETWORK_NAME"
    psk="YOUR_PASSWORD"
    key_mgmt=WPA-PSK
}
```
Write file by pressing Ctrl+W, and close it by Ctrl+X.
After the first boot, Linux will move this file to /etc/wpa_supplicant/ folder.

5) For ssh connection navigate to /boot file
```c
cd /boot
```

Create empty file ssh

```c
touch ssh
```

Restart Raspberry Pi.

6) Enable WiFi Hotspot on Windows

![](images/mob_hotspot2.jpg))

Connected devices IP addresses can be viewed from the table below. Currently, the maximum number of devices is 8.

7) Download PyTTY (https://www.putty.org/ ) for SSH connections

In the main menu enter the IP address of the Raspberry Pi, for connection type select SSH and click Open:

![](images/RaspberryPi_connection_putty.png)

8) Enter login and password for access. The default login is “pi”, and password is “raspberry”

![](images/RaspberryPi_connection.png)

#### Camera Calibration and code explanation
Any camera that you will be using, it needs to be calibrated as well. In case of using a raspberry pi on the drone, it is better to use the raspberry camera, because of its lightweight and easy connection to the Pi 

For calibrating the camera you will be required OpenCV Library, it is one of the most widely used packages for video detection, motion detection, image recognition, and deep learning.

Follow the codes one by one to install and compile OpenCV 4.1.2 on Raspbian OS

    $ chmod +x *.sh
    $ ./download-opencv.sh
    $ ./install-deps.sh
    $ ./build-opencv.sh
    $ cd ~/opencv/opencv-4.1.2/build
    $ sudo make install

The full description of code lines can be found in the [here](https://gist.github.com/willprice/abe456f5f74aa95d7e0bb81d5a710b60)

After installing OpenCV, do a test run by using below codes

    $ wget "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png" -O lenna.jpg
    $ python2 test.py lenna.jpg
    $ python3 test.py lenna.jpg

Now, for detecting an Aruco marker by a Raspberry Pi camera 
Follow the procedure:- 

1. Download the checkboard [image](https://github.com/HSRW-EMRP-WS1920-Drone/object_detection_and_landing/blob/master/opencv/ChessBoard_9x6.jpg) and take a printout

2.  Now you have to hold that checkboard printout image and take pictures using the raspberry pi camera.

For taking images, run [save_images.py](https://github.com/HSRW-EMRP-WS1920-Drone/object_detection_and_landing/blob/master/opencv/save_images.py) script: 

`$ python save_images.py`

(Press spacebar for saving images)
**The images will be saved in the same folder where the (camera_test.py) python script is located**
Take almost 30 to 35 images of the checkboard and save in Raspberry Pi memory.

3 . Images processing

After images are collected, they are processed by [cameracalib.py](https://github.com/HSRW-EMRP-WS1920-Drone/object_detection_and_landing/blob/master/opencv/cameracalib.py) file.

Command to run

    cameracalib.py  <folder> <image type> <num rows> <num cols> <cell dimension>

there 
folder - path containing images files, default "./camera_01"
image type - the type of these images, default "jpg"
num_rows - number of rows, default 9
num_cols - number of cols, default 6
cell dimension - dimension of cell in mm, default 25

Code explained

1) Do not process if the number of images is less than 9


```python
if len(images) < 9:
    print("Not enough images were found: at least 9 shall be provided!!!")
    sys.exit()
```

2) For each image in folder find chessboard corners with findChessboardCorners method and draw them with drawChessboardCorners method from OpenCV. At the same time, store object points (ids of eacht image) and corners  in an array:
    
        nPatternFound = 0
        imgNotGood = images[1]
    
        for fname in images:
            if 'calibresult' in fname: continue
            #-- Read the file and convert in greyscale
            img     = cv2.imread(fname)
            gray    = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
            print("Reading image ", fname)
    
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (nCols,nRows),None)
    
            # If found, add object points, image points (after refining them)
            if ret == True:
                print("Pattern found! Press ESC to skip or ENTER to accept")
                #--- Sometimes, Harris cornes fails with crappy pictures, so
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nCols,nRows), corners2,ret)
                cv2.imshow('img',img)
                # cv2.waitKey(0)
                k = cv2.waitKey(0) & 0xFF
                if k == 27: #-- ESC Button
                    print("Image Skipped")
                    imgNotGood = fname
                    continue
    
                print("Image accepted")
                nPatternFound += 1
                objpoints.append(objp)
                imgpoints.append(corners2)
    
                # cv2.waitKey(0)
            else:
                imgNotGood = fname
        cv2.destroyAllWindows()
    
3) if any pattern was found, calculate distortion coefficient and camera matrix with calibrateCamera method

        if (nPatternFound > 1):
            print("Found %d good images" % (nPatternFound))
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        
            # Undistort an image
            img = cv2.imread(imgNotGood)
            h,  w = img.shape[:2]
            print("Image to undistort: ", imgNotGood)
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        
            # undistort
            mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
            dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
        
            # crop the image
            x,y,w,h = roi
            dst = dst[y:y+h, x:x+w]
            print("ROI: ", x, y, w, h)
        
            cv2.imwrite(workingFolder + "/calibresult.png",dst)
            print("Calibrated picture saved as calibresult.png")
            print("Calibration Matrix: ")
            print(mtx)
            print("Disortion: ", dist)

4) Save results to txt files

        filename = workingFolder + "/cameraMatrix.txt"
        np.savetxt(filename, mtx, delimiter=',')
        filename = workingFolder + "/cameraDistortion.txt"
        np.savetxt(filename, dist, delimiter=',')
        mean_error = 0
        for i in xrange(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx,             dist)
            error = cv2.norm(imgpoints[i],imgpoints2,             cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        print("total error: ", mean_error/len(objpoints))

Calibration computes a set of camera indicators (camera matrix, distortion) for a specific camera and should be done only once unless the camera's optic characteristics change.


### Drone part

Select the drone as per your requirement, after selecting the type of drone-based on the motors or propellers, weather it is quadcopter or hexacopter it has to be calibrated.
It can be calibrated with the help of mission planner software
To calibrate follow the steps one by one: -
1. Download and install the mission planner software from here (https://firmware.ardupilot.org/Tools/MissionPlanner/).
2.     Click on INITIAL SETUP > Install Firmware > (based on your number of motors in drone select the ArduCopter version)
3.     After connecting from drone pixhawk to your laptop or pc click on CONNECT in mission planner
4.     Go to INITIAL SETUP click on Frame type (select the frame according to your drone).
5.     Followed to that in INITIAL SETUP click on mandatory hardware for calibrating: -
         i. Accelerometer Calibration
        ii. Compass Calibration
          iii. Radio Calibration

i. Accelerometer Calibration:
For a drone to let know all the sides and directions, accelerometer calibration is required, it helps the drone to identify which side is left, right front and back.
1. Click Accelerometer Calibration to start the calibration, then click Calibrate Accel. 
2. Go according to the Guidelines of Calibrate Accel of setting sides of your drone. The setup will take around 5 to 10 mins.
3. In case required full instructions, please visit the website:-
https://ardupilot.org/copter/docs/common-accelerometer-calibration.html

![acc](images/drone_1.png)
https://ardupilot.org/copter/docs/common-accelerometer-calibration.html

ii. Compass Calibration: -
It is for the drone to understand all the directions.
1. To perform the onboard calibration of the compass on the drone:
2. Click on Compass in Mandatory hardware
3. Select the device Pixhawk/PX4, then click start, you will hear a beep sound once per second. This means the time has started to rotate the drone.
4. Hold the drone and rotate it in all the directions like a full rotation from the front, back, left, right, top and bottom and keep rotating until you hear three rising beeps, which means your calibration is successful. If you hear an unhappy failure tone, start the procedure again from step 3.
5. In case required instruction in detail, please visit the website: https://ardupilot.org/copter/docs/common-compass-calibration-in-mission-planner.html

![comp](images/drone_2.png)
https://ardupilot.org/copter/docs/common-compass-calibration-in-mission-planner.html

iii. Radio Calibration
It is the calibration between controller and drone, to set the proper connection of the transmitter and the receiver in both the devices.

• Click on Radio Calibration in mandatory hardware, then click Calibrate Radio on the bottom right, press “OK”, when prompted to check the radio control equipment, is on, the battery is not connected, and propellers are not attached.
![radio](images/drone_3.jpg)
ttps://ardupilot.org/copter/docs/common-radio-control-calibration.html

•  Follow the complete instruction in detail given on the website: https://ardupilot.org/copter/docs/common-radio-control-calibration.html
Select Click when Done
• A window will appear with the prompt, “Ensure all your sticks are centered and the throttle is down and click ok to continue”. Move the throttle to zero and press “OK”.
• Mission Planner will show a summary of the calibration data. Normal values are around 1100 for minimums and 1900 for maximums.

![radio](images/drone_4.jpg)
https://ardupilot.org/copter/docs/common-radio-control-calibration.html

### Raspberry Pi to Pixhawk connection
#### Hardware Communication
Communication is done through the Telem2 telemetry port of Pixhawk. Ground, Rx (receiver), Tx(transmitter) and +5V should be connected to 4 pins of Raspberry Pi.

![](images/f837b6b1116ec02c3490e34035c2f09da5a62936.jpg)

The required packages should be installed

```c
sudo apt-get update    #Update the list of packages in the software center
sudo apt-get install python3-dev python3-opencv python3-wxgtk3.0 libxml2-dev python3-pip python3-matplotlib python3-lxml
sudo pip3 install future
sudo pip3 install pymavlink
sudo pip3 install mavproxy
```

#### Configuring serial port

Type `sudo raspi-config`

In the configuration window select “Interfacing Options”:

![](images/RaspberryPi_Serial1.png)


And then “Serial”:

Answer “No” to “Would you like a login shell to be accessible over serial?” and “Yes” to 
“Would you like the serial port hardware to be enabled?” questions.

#### Testing connection
![](images/RaspberryPi_Serial2.png)

Run mavlink.py file to test the connection with Pixhawk

```c
sudo -s 
mavproxy.py --master=/dev/ttyAMA0 --baudrate 57600 --aircraft MyCopter
``` 

In the case of a successful connection, the output should be the following

![](images/RaspberryPi_ArmTestThroughPutty.png)

### Landing with aruco marker approach
#### Aruco marker

An ArUco marker is a black square marker with the inner binary representation of the identifier. Having black borders, these markers are easy to detect in a frame.

![aruco](images/aruco_markers.jpg)
> Aruco Marker Samples

For each the specific application a dictionary – a set of markers – is defined. Dictionaries have such properties as the dictionary size and the marker size. The size of the dictionary is defined by the number of markers it contains, and the marker size is the number of bits it has in the inner part.
The identification code of the marker is not the result of the conversion of a binary image to a decimal base, but the market index in the dictionary. The reason is that for a high number of bits the results may become unmanageable. 

#### Marker creation
OpenCV library provides methods to create and detect aruco markers. The drawmarker() method is defined for generating markers. Dictionary should be chosen beforehand. Example:

```c
cv::Mat markerImage;
cv::Ptr<cv::aruco::Dictionary> dictionary cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
cv::aruco::drawMarker(dictionary, 23, 200, markerImage, 1);
cv::imwrite("marker23.png", markerImage);`
```

Parameters:
-      Dictionary object, created from getPredefinedDictionary method
-      marker id – should be in the range, defined for the selected dictionary
-      the size of the marker in pixels
-      output image
-      black border width, expressed by the number of internal bits (optional, default equals 1)

There are also online aruco generators, one of which was used in this project. As an example, http://chev.me/arucogen/ website can be mentioned.

#### Marker detection

The goal of marker detection is to return the position and id of each marker found in the image. This process can be divided into two steps:

1.  Possible candidates’ detection – return all the square shapes and discard non-convex ones, by analyzing contours of each figure.
2.  For each candidate, its inner codification is analyzed. Several actions are taken as:
    • transformation to canonical form
    • black and white bits are separated
    • division to cells following marker size
    • the number of black and white pixels is counted to determine the color of a cell
    • the bits are analyzed to their relevance to the selected dictionary

Results of detection can be visualized

![aruco](images/singlemarkersdetection.jpg)
>Detected markers

![aruco](images/singlemarkersrejected.jpg)
>Rejected markers


detectMarkers() function from the aruco module is used for detection. Parameters:
    • image with markers for detections
    • the dictionary object
    • structure of marker corners
    • list of marker ids
    • DetectionParameters class object – used for customizing detection parameters
    • list of rejected parameters 

Example:

```c
cv::Mat inputImage;
std::vector<int> markerIds;
std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
cv::aruco::detectMarkers(inputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
```

The found markers can be drawn using drawDetectedMarkers(). Example:
```c
cv::Mat outputImage = inputImage.clone();
cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);
```

#### Camera Pose Estimation

Camera pose can be obtained from the position of the markers if the camera was calibrated (camera matrix and distortion coefficients are available). The camera pose is the transformation from the marker coordinate system to the camera coordinate system. Rotation and transformation are estimated in the form of vectors.

```c
cv::Mat cameraMatrix, distCoeffs;

std::vector<cv::Vec3d> rvecs, tvecs;
cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);
```
There rvecs and tvecs are rotation and transformation vehicles respectively.

#### Landing algorithm and code explanation

The algorithm steps:

1) check the aruco class object for new detection. ArucoSingleTracker is a wrapper class for access to aruco API.

```c
marker_found, x_cm, y_cm, z_cm = aruco_tracker.track(loop=False) 
```

loop = False – do detection once.

Output:
marker_found – flag, indicating that the marker was found, boolean
x_cm – x coordinate of the marker on the image
y_cm – y coordinate of the marker on the image
z_cm – z coordinate of the marker on the image, for the images taken from less than 5 meters, z_cm is taken as an altitude of the drone.

2) If marker_found is True convert x and y coordinates from camera coordinate system to drone coordinate system. The formula is

![aruco](images/coords_to.png)
![aruco](images/coords_to2.png)

where ![aruco](formulas/f1.jpg) are board frame coordinates, ![aruco](formulas/f2.jpg) are camera frame coordinates.

3) A drone can navigate to the location of the marker with or without altitude reduction. The decision is done according to the angle between the drone's vertical axis and the vector to the marker. This angle indicates drones closeness to the marker and commands to move with a landing when it is less than some threshold values. The command  of moving with descending is given if the expression

 ![aruco](formulas/f3.jpg)
 
returns true value, where  ![aruco](formulas/f4.jpg) and  ![aruco](formulas/f5.jpg) in radians.

4) Calculate latitude and longitude from the marker coordinates. The algorithm was taken from the gis portal and is relatively accurate over small distances (10m within 1km). First, north and east attitudes should be calculated for the current yaw of a drone. Yaw is a rotation indicator in horizontal space.

 ![aruco](images/Yaw_Axis_Corrected.svg.png)

 ![aruco](formulas/f6.jpg)
 
  ![aruco](formulas/f7.jpg)

Second, latitude and longitude are calculated. The drone’s coordinate is taken from GPS, earth radius is taken approximately 6378137 meters.

![aruco](formulas/f8.jpg)

![aruco](formulas/f9.jpg)

![aruco](formulas/f10.jpg)

![aruco](formulas/f11.jpg)

5) Check angles, calculated at step 3 :	go to the marker location at the same altitude or go with the lowering down. Dronekit's simple_goto function for navigating vehicle to specified latitude and longitude is used.

![landing](images/gowithoutlowering.png)

![landing](images/gowithlowering.png)

![landing](images/godown.png)

6) If the height of a drone is less than some threshold altitude, perform vertical landing by changing the mode of the vehicle to “LAND” value.

```c
vehicle.mode = "LAND" 
```
#### Landing algorithm code explained

Further, [aruco_landing.py](https://github.com/HSRW-EMRP-WS1920-Drone/object_detection_and_landing/blob/master/scripts/aruco_landing.py) script containing landing algorithm is detailly explained.

1) Import libraries from dronekit, pymavlink packages for connection with a drone

DroneKit-Python is an open-source project, which provides APIs for communication with a vehicle over MAVlink protocol. The aim of these APIs is reading the vehicle's state and parameters, and also enables control over drone's movements. 

The main classes and method used in this project are:

connect() - creates Python object for accessing drone's parameters and commands
Example:
```python
connection_string = "/dev/ttyAMA0"
baud_rate = 57600
vehicle = connect(connection_string,baud=baud_rate, wait_ready=True)  
```

LocationGlobalRelative(<latitude>, <longitude>, <altitude>) - creates new location object from given latitude, longitude and altitude, which is then passed to navigation commands

simple_goto(<location>) - command of dronekit object (*vehicle* from above) for navigating to given location

```python
from drone kit import connect, VehicleMode, LocationGlobalRelative, Command, LocationGlobal
from pymavlink import mavutil
```

2) import wrapper class for using Aruco methods
```python
from opencv.lib_aruco_pose import *
```
3) function definitions

*get_location_metres*
Arguments:

original_location - object containing drone latitude and longitude
dNorth, dEast - North and East attitude of drone, real value

Returns object containing new latitude and longitude.

```python
from opencv.lib_aruco_pose import *
```
```python
    
#--------------------------------------------------
#-------------- FUNCTIONS  
#--------------------------------------------------    

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
    
    print "dlat, dlon", dLat, dLon

    #New position in decimal degrees
    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    return(newlat, newlon)
```
*marker_position_to_angle*
Arguments:

x,y,z coordinates of detected marker

Returns angles between x and z,y and z.

```python
def marker_position_to_angle(x, y, z):
    
    angle_x = math.atan2(x,z)
    angle_y = math.atan2(y,z)
    
    return (angle_x, angle_y)
```
*camera_to_uav*
Arguments:
x,y coordinates of detected marker
Returns converted x,y coordinates of marker to uav coordinate system
```python
def camera_to_uav(x_cam, y_cam):
    x_uav =-y_cam
    y_uav = x_cam
    return(x_uav, y_uav)
```
*uav_to_ne*
Arguments:
x,y coordinates, yaw angle coordinates of drone
Returns north and east attitudes of the drone

```python    
def uav_to_ne(x_uav, y_uav, yaw_rad):
    c       = math.cos(yaw_rad)
    s       = math.sin(yaw_rad)
    north   = x_uav*c - y_uav*s
    east    = x_uav*s + y_uav*c 
    return(north, east)
```
*check_angle_descend*
Arguments:
angle_x,angle_y, angle_desc angles
Compares angles x and y with a threshold value of angle_desc and returns
boolean true, commanding to descend and false, commanding to move at the same altitude.

```python    
def check_angle_descend(angle_x, angle_y, angle_desc):
    return(math.sqrt(angle_x**2 + angle_y**2) <= angle_desc)
```
4) connecting to drone
```python
#--------------------------------------------------
#-------------- CONNECTION  
#--------------------------------------------------    
#-- Connect to the vehicle
print('Connecting...')
connection_string = "/dev/ttyAMA0"
vehicle = connect(connection_string,baud=baud_rate, wait_ready=True)  
```
5) constant variables for algorithm definded
```python
#--------------------------------------------------
#-------------- PARAMETERS  
#-------------------------------------------------- 
rad_2_deg   = 180.0/math.pi
deg_2_rad   = 1.0/rad_2_deg 

#--------------------------------------------------
#-------------- LANDING MARKER  
#--------------------------------------------------    
#--- Define Tag
id_to_find      = 72
marker_size     = 10 #- [cm]
freq_send       = 1 #- Hz

land_alt_cm         = 50.0
angle_descend       = 20*deg_2_rad
land_speed_cms      = 30.0


```
6) camera matrix and distortion are loaded
```python
#--- Get the camera calibration path
# Find full directory path of this script, used for loading config and other files
cwd                 = path.dirname(path.abspath(__file__))
calib_path          = cwd+"/../opencv/"
camera_matrix       = np.loadtxt(calib_path+'cameraMatrix_raspi.txt', delimiter=',')
camera_distortion   = np.loadtxt(calib_path+'cameraDistortion_raspi.txt', delimiter=',')                                      
aruco_tracker       = ArucoSingleTracker(id_to_find=id_to_find, marker_size=marker_size, show_video=False, 
                camera_matrix=camera_matrix, camera_distortion=camera_distortion)
                
                
time_0 = time.time()

while True:                
```
7) aruco tracker object is commanded to track marker. It returns marker_found flag, x,y,z coordinates of the marker 
```python
    marker_found, x_cm, y_cm, z_cm = aruco_tracker.track(loop=False)
    if marker_found:
        x_cm, y_cm          = camera_to_uav(x_cm, y_cm)
        uav_location        = vehicle.location.global_relative_frame
        
```
8) if the drone's altitude is higher than 5 meters, use drone altitude as z coordinate
```python
        #-- If high altitude, use baro rather than visual
        if uav_location.alt >= 5.0:
            print 
            z_cm = uav_location.alt*100.0
            
        
```
9) calculate angle x and angle y using marker_position_to_angle method
```python
        angle_x, angle_y    = marker_position_to_angle(x_cm, y_cm, z_cm)

```
10) check that a second passed after the last marker detection to prevent too frequent processing
```python       
        if time.time() >= time_0 + 1.0/freq_send:
            time_0 = time.time()
            # print ""
            print " "
            print "Altitude = %.0fcm"%z_cm
            print "Marker found x = %5.0f cm  y = %5.0f cm -> angle_x = %5f  angle_y = %5f"%(x_cm, y_cm, angle_x*rad_2_deg, angle_y*rad_2_deg)
```
11) calculate north and east attitude
```python       
            north, east             = uav_to_ne(x_cm, y_cm, vehicle.attitude.yaw)
            print "Marker N = %5.0f cm   E = %5.0f cm   Yaw = %.0f deg"%(north, east, vehicle.attitude.yaw*rad_2_deg)
```
12) calculate marker's latitude and longitude
```python       
            marker_lat, marker_lon  = get_location_metres(uav_location, north*0.01, east*0.01)  
            #-- If angle is good, descend
```
13) compare angle to decide about descending
```python      
            if check_angle_descend(angle_x, angle_y, angle_descend):
                print "Low error: descending"
                location_marker         = LocationGlobalRelative(marker_lat, marker_lon, uav_location.alt-(land_speed_cms*0.01/freq_send))
            else:
                location_marker         = LocationGlobalRelative(marker_lat, marker_lon, uav_location.alt)
                
```
14) call  *simple_goto* method to navigate to a new location 
```python      
            vehicle.simple_goto(location_marker)
            print "UAV Location    Lat = %.7f  Lon = %.7f"%(uav_location.lat, uav_location.lon)
            print "Commanding to   Lat = %.7f  Lon = %.7f"%(location_marker.lat, location_marker.lon)
            
```
15) compare altitude to judge whether land or not
```python 
        #--- COmmand to land
        if z_cm <= land_alt_cm:
            if vehicle.mode == "GUIDED":
                print (" -->>COMMANDING TO LAND<<")
                vehicle.mode = "LAND"
```

### Troubleshooting
#### Troubleshooting: Raspberry Pi Camera
Sometimes the camera does not get detected so use the following commands to solve the problem:
(Check if the camera is supported and detected) 
```c
vcgencmd get_camera
```
-supported=1 detected=1 (detected=0 means no camera detected)
```c
sudo raspi-config
```
>Interfacing Options Configure connections to peripherals
>P1 Camera Enable/Disable (Enable the camera manually)
Then reboot the Raspberry Pi


#### Troubleshooting: Run the python script at bootup

You can run the python script at bootup with Crontab. We were running the python script in the testing phase with remote access. It is better to enable automatic execution of python script at a specific time of the day or when the Raspberry Pi boots up. First, we need to run the code:

Run crontab with the -e flag to edit the cron table:

```c
crontab -e
```

Then type the following command in the crontab:

    0 8 * * 1-5 python /home/pi/code1.py
    # * * * * *  command to execute
    # ┬ ┬ ┬ ┬ ┬
    # │ │ │ │ │
    # │ │ │ │ │
    # │ │ │ │ └───── day of the week (0 - 7) (0 to 6 are Sunday to Saturday, or use names; 7 is Sunday, the same as 0)
    # │ │ │ └────────── month (1 - 12)
    # │ │ └─────────────── day of the month (1 - 31)
    # │ └──────────────────── hour (0 - 23)
    # └───────────────────────── min (0 - 59)


To run a command every time the Raspberry Pi starts up, write @reboot instead of the time and date. For example:

```c
@reboot python /home/pi/video_with_timestamp.py
```

Source link to read in detail: https://www.raspberrypi.org/documentation/linux/usage/cron.md
https://crontab-generator.org/

For example:
If you want to run the code every 10 Minutes, between 8 AM and 8 PM, from Monday to Friday or have a task that should only run during normal business hours, then this could be accomplished using ranges that for the hour and weekday values separated by a dash.

In other words: “At every 10th minute past every hour from 8 through 20 on every day-of-week from Monday through Friday”


`*/10 8-20 * * 1-5`


You can edit your cron jobs with:

```c
crontab -l
```

You can cancel it with:

```c
crontab -r
```

link: https://fireship.io/snippets/crontab-crash-course/



### Computer Vision


#### How do we build a Deep Learning model

•Step 1: Collect Images

•Step 2: Label the images

•Step 3: Finding a Pretrained Model for Transfer Learning

•Step 4: Training on a GPU (cloud service like AWS/Google Colab etc or GPU Machine)

•Step 5: Test the model via the device

•Step 6: Device predicts objects in the image or frame


![darknet](images/dlmodel.png)

https://cdn-images-1.medium.com/max/1600/1*huoie8skkgmqx68-279z_a.jpeg?fbclid=iwar0qvjmvg9ukllowedwkuxqsfjwlmoaeijaxmgrucylulfcjy4sdtg-6rkm
### Yolo implementation
We choose YOLO which is the fastest object detection algorithm available for object detection.
We choose Yolov3-tiny architecture for training.

Main problems of object detection on Raspberry Pi:
-YoloV3 has higher resource consumption that Raspberry Pi (3B) can offer
-Absence of GPU on Raspberry Pi for image processing
-YoloV3 processes input frames at 30 fps on a powerful GPU but does not work on Pi

Suggested solution:
-Using YoloV3 tiny version (with degraded accuracy, but better performance)
-Implementing NNPACK acceleration library (for parallel computation on multi-core CPU)
-The small architecture of Yolov3-tiny with 270 fps speed
-A trade-off between speed and accuracy (high speed and low accuracy than YOLOv3)

As we have no GPU, we decided to train the model on Google Colab.

#### Data Collection

![darknet](images/data_col.png)


Data was needed for training the tiny yolo but unfortunately, the images from the internet were unacceptable because the model works better if the inference images are as close as possible to the training images. We took six videos in different light intensity and background as shown in the above picture. The object will be under the drone so the banana was put on the ground. Sometimes there were multiple bananas in the video. Frames were extracted from the video using the following tool in Windows:
https://www.chip.de/downloads/Free-Video-to-JPG-Converter_30220246.html
One frame was extracted from each second. In total there were 1800 images of banana on the ground.

#### Data Annotation

![darknet](images/data_ann1.png)
![darknet](images/data_ann2.png)
![darknet](images/data_ann3.png)
![darknet](images/data_ann4.png)

We had to manually draw a bounding box for 1800 pictures of Banana like the above pictures. We used the following tool given by Ivangrob to do so:
Using OpenLabeling (https://github.com/ivangrov/YOLOv3-Series) tool bananas were marked on images.

As a result we obtain the same number of text documents with four variables:
<class_id> <x> <y> <width> <height>

![darknet](images/data_ann5.png)

#### GPU Support

![darknet](images/colab.png)


We choose Google colab over AWS and Floydhub for the following reasons:

●Provides free GPU support

●Allows developers to use and share Jupyter Notebooks among each other like Google Docs

●Can access data from Google Drive

●All major Python libraries, like TensorFlow, Scikit-learn, Matplotlib, etc. are pre-installed

●Built on top of Jupyter Notebook

●Run two files for 12 hours


#### Training tiny YOLO v3 in Google Colab
To train our banana model we used open-source neural network framework darknet (https://pjreddie.com/darknet/). (AlexyAB GitHub instructions)

Benefits:
- Fast and Easy to install
- CPU and GPU computation support

##### STEP 1. Connect the Colab notebook to Google Drive
One advantage of using Google Colab is that the data can be accessed directly from the google drive. We have to mount the drive on the Colab, then Colab will send the link for authorization code and offer input for this code typing. If drive was already mounted, to force remount of google drive, the following parameter should be added to mount method
force_remount=True

```python
drive.mount('/content/gdrive', force_remount=True)
```
 
```python
# These commands import the drive library and mount your Google Drive as a local drive. You can access your Drive files using this path "/content/gdrive/My Drive/"
 
from google.colab import drive
drive.mount('/content/gdrive')
 
#for force remount of drive
#drive.mount('/content/gdrive',force_remount=True)
```

##### STEP 2. Check CUDA release version

Before installing cuDNN CUDA version should be checked. 

    # CUDA: Let's check that Nvidia CUDA is already pre-installed and which version is it
    !/usr/local/cuda/bin/nvcc --version


##### STEP 3. Install cuDNN according to the current CUDA version

The version of cuDNN for linux (archived as tgz) was downloaded from https://developer.nvidia.com/ and uploaded to google drive.


    # We're unzipping the cuDNN files from your Drive folder directly to the VM CUDA folders
    !tar -xzvf gdrive/My\ Drive/darknet/cuDNN/cudnn-10.0-linux-x64-v7.5.0.56.tgz -C /usr/local/
    !chmod a+r /usr/local/cuda/include/cudnn.h
    
    # Now we check the version we already installed. Can comment this line on future runs
    !cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
    

##### STEP 4. Installing Darknet

Darknet was cloned from https://github.com/kriyeng/darknet/ repository and compiled.


    !git clone https://github.com/kriyeng/darknet/
    %cd darknet    
     
    # Check the folder
    !ls
     
    # I have a branch where I have done the changes commented above
    !git checkout feature/google-colab
     
    #Compile Darknet
    !make
    
    #Change the mode of darknet file, so it can be execut
    !chmod +x ./darknet

This should be done only once, then it is saved in google drive.

##### Some Utils

Let's add some utils that maybe can be useful.
These utils are:
imgShow() - Will help us to show an image in the remote VM 
download() - Will allow you to get some file from your notebook in case you need to 
upload() - You can upload files to your current folder on the remote VM
 
```python
def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
  %matplotlib inline
 
  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)
 
  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  #plt.rcParams['figure.figsize'] = [10, 5]
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()
def upload():
  from google.colab import files
  uploaded = files.upload() 
  for name, data in uploaded.items():
    with open(name, 'wb') as f:
      f.write(data)
      print ('saved file', name)
def download(path):
  from google.colab import files
  files.download(path)
```
. 
To test that darknet was correctly installed, the following code can be run

```python
# Execute darknet using YOLOv3 model with pre-trained weights to detect objects on 'person.jpg'
!./darknet detect cfg/yolov3.cfg yolov3.weights data/person.jpg -dont-show
 
# Show the result using the helper imgShow()
imShow('predictions.jpg')
```
##### PART 2. Training YOLO
##### STEP 1. Preparing your data and configuration files
 
Upload img folder with images and txt files to google drive /darknet/img
create obj.data and obj.names files
obj.data (count of classes, paths to train, test, class names and results folders) content:


    classes= 1
    train = /content/gdrive/My\ Drive/darknet/train.txt
    valid = /content/gdrive/My\ Drive/darknet/test.txt 
    names = /content/gdrive/My\ Drive/darknet/obj.names
    backup = /content/gdrive/My\ Drive/darknet/backup

 
obj.names (names of classes) content:
```python
Banana
```
 
##### List all jpg files
```python
import os
directory = "/content/gdrive/My Drive/darknet/img"
filelist=os.listdir(directory)
for fichier in filelist[:]:# filelist[:] makes a copy of filelist.
  if not(fichier.endswith(".jpg")):
    filelist.remove(fichier)
print(filelist)
Divide the list of files to training and testing set (80% and 20%)
train = open('/content/gdrive/My Drive/darknet/train.txt', 'w')
test=open('/content/gdrive/My Drive/darknet/test.txt', 'w')
len_of_list=len(filelist)
eighty_limit=0.8*len_of_list
iterator=1
for item in filelist:
  if(iterator > eighty_limit):
    test.write("/content/gdrive/My Drive/darknet/img/%s\r\n" % item)
  else:
    train.write("/content/gdrive/My Drive/darknet/img/%s\r\n" % item)
  iterator=iterator+1
 
train.close()
test.close()
```

Copy yolov3-tiny.weights.cfg to yolo-obj.cfg) and change:

batch=64 

subdivisions=16

![darknet](images/darknet_conf1.jpg)

Filters field  in each [convolutional] block befor [yolo] should be changed in accordance with equation:

`filters=(classes + 5)x3`

So, for 1 class there should be filters=18, for 2 classes filters=21.

![darknet](images/darknet_conf2.jpg)

##### Calculate Anchor and update the values in the anchor file

`!./darknet detector calc_anchors data/obj.data -num_of_clusters 6 -width 608 -height 608
`
The output is an array of anchors

`anchors =  52,144,  94, 92,  81,179, 155,156, 117,303, 245,303`

They should be put into anchors field of each [yolo] block of the cfg file.

![darknet](images/darknet_conf3.jpg)


##### Initial weights file preparation
Weights file is required for training model in darknet. This enables *continuing* training from the point where algorithm stopped, as darknet saves weights file after every 1000 iterations. But for the first run of command there is not yet any file generated. In this case standard weights file "yolov3-tiny.weights" may be used. Another possible solution is to generate an initial weights file by running the command

`!./darknet partial yolov3-tiny0802.cfg yolov3-tiny.weights weights/yolov3-tiny.conv.15 15`

which will save the file to "yolov3-tiny.conv.15" in this case.

##### Training model

To train model, the following command should be run in Linux (Google Colab implementation)

`!./darknet detector train <data> <config> <weights> -dont_show -map`

, where 
- data – is a path to obj.data file (path to the image and backup folder),
- config is a path to the configuration file(hyperparameter tuning),
- weights is a path to initial weights file
- -dont_show flag tells the program to suppress display output
- -map tells to generate mAp (mean average precision) diagram

`!./darknet detector train data/obj.data yolov3-tiny0802.cfg weights/yolov3-tiny.conv.15 -dont_show -mjpeg_port 8090 -map`

##### When should I stop training

Usually, one class requires about 2000 iterations, and 4000 iterations is a minimum for all classes.
Darknet outputs average error on each iteration. Stop training when no longer avg loss decreases.

When training is over, different weight files (for each thousand iteration count) in /darknet/backup folder can be used in the detection and the best one is chosen. Using weights file with maximum iterations is not always the best solution for the **Overfitting** phenomena when the model is trained too much and can recognize only the objects, which look exactly like the training set images.

A copy of the file with beforementioned operations was saved from Google Colab as ipynb file in this repository, and can be accessed from [here](https://github.com/HSRW-EMRP-WS1920-Drone/object_detection_and_landing/blob/master/darknet%20Google%20Colab/Tiny_yolo_v3_explained.ipynb).

For more information, visit the following links:

**Tutorial Darknet To Colab**
https://colab.research.google.com/drive/1lTGZsfMaGUpBG4inDIQwIJVW476ibXk_#scrollTo=Cqo1gtPX6BXO 
**AlexeyABs Darknet repo**
https://github.com/AlexeyAB/darknet/#how-to-train-tiny-yolo-to-detect-your-custom-objects
**YOLOv3 Series by Ivan Goncharov**
https://www.youtube.com/watch?v=R0hipZXJjlI&list=PLZBN9cDu0MSk4IFFnTOIDihvhnHWhAa8W 
**Ivangrovs Darknet repo**
https://github.com/ivangrov/YOLOv3-Series


##### How to calculate mAP
Mean Average Precision (mAP) chart can be calculated while the model is learning. To do this in training command *-map* parameter should be added:

`darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -map`

##### How to improve object detection:

in .cfg-file to increase precision

- set flag random=1

- increase network resolution (height=608, width=608 or any value multiple of 32)

Also 

- check that each object is labeled in the dataset

- run training with `-show_imgs` argument to see bounding boxes

- include into dataset images with many other objects and images without the target object

#### Tiny YOLOv3 results for banana
##### Mean Average Precision for our model

![map](images/map.png)

The above picture depicts the mean average precision values for every 1000 iterations of our models. We reached mAP50 96% which is very good. We can take 2000, 3000 or 4000 weights and see which one is performing better. We finally choose the weight for 4000 iterations after lots of testing because the Bananas were recognized best.

To learn more about Mean Average Precision(mAP), visit the following link:
Precision, Recall & Mean Average Precision for Object Detection
https://www.youtube.com/watch?v=e4G9H18VYmA&list=PL1GQaVhO4f_jE5pnXU_Q4MSrIQx4wpFLM




#### NNPack
Even though YoloV3 is efficient, it still requires computational power from the device it operates on. A simplified version of Yolo - Yolov3-tiny - runs faster but detects less accurately. Both these approaches didn't find the object when tested on Raspberry Pi.
A decision was made to use NNPACK - a library for the neural network to run on a multi-core CPU. The following steps should be completed to use NNPack.

**Step 0**
Prepare Python and camera.

Install  package installer for Python *pip*

`sudo apt-get install python-pip`

Enable camera in Raspberry Pi. For it run code:

`sudo raspi-config`

Navigate to `Interfacing Options`, and enable `Camera`
To be able to use a camera, a reboot is required.

**Step 1**
**NNPack installation**

NNPack is used to optimize Darknet for embedded devices with ARM CPUs without using a GPU. It uses transform-based convolution computation which allows 40% faster performance on non-initial frames. Particularly, for repeated inferences, ie. video, NNPack is more beneficial to use.

**Install building tool Ninja**

Install [PeachPy](https://github.com/Maratyszcza/PeachPy) and [confu](https://github.com/Maratyszcza/confu)

    sudo pip install --upgrade git+https://github.com/Maratyszcza/PeachPy
    sudo pip install --upgrade git+https://github.com/Maratyszcza/confu

Install [Ninja](https://ninja-build.org/)

    git clone https://github.com/ninja-build/ninja.git
    cd ninja
    git checkout release
    ./configure.py --bootstrap
    export NINJA_PATH=$PWD
    cd

Install [NNPack](https://github.com/shizukachan/NNPACK)

    git clone https://github.com/shizukachan/NNPACK
    cd NNPACK
    confu setup
    python ./configure.py --backend auto

Build NNPack using Ninja

`$NINJA_PATH/ninja
`
Run `ls` to view files, lib and include folders should be listed

`ls`

Copy the libraries and header files to the system environment:

    sudo cp -a lib/* /usr/lib/
    sudo cp include/nnpack.h /usr/include/
    sudo cp deps/pthreadpool/include/pthreadpool.h /usr/include/

**Step 2**
**Install darknet-nnpack**

    cd
    git clone -b yolov3 https://github.com/zxzhaixiang/darknet-nnpack
    cd darknet-nnpack
    git checkout yolov3
    make

**Step 3**
**Testing with YoloV3-tiny**

To test NNPack, 2 files are included: rpi_video.py and rpi_record.py. The former only displays the video, while the latter also records each frame. They can be run by commands:

`sudo python rpi_video.py`

or

`sudo python rpi_record.py`

Weight and configuration file paths can be changed inside beforementioned python files at the line:

    yolo_proc = Popen(["./darknet",
                       "detect",
                       "./cfg/yolov3-tiny.cfg",
                       "./yolov3-tiny.weights",
                       "-thresh", "0.1"],
                       stdin = PIPE, stdout = PIPE)

##### Result
![nnpack](images/giphy.gif)

#### Testing the model
The next step is to do inference or test the YOLO model. The trained weight of the model was download from Google Colab. 
For inference, we first used the code provided by ivangrov. His code is also uploaded to this Github. Here is the [link](https://github.com/HSRW-EMRP-WS1920-Drone/object_detection_and_landing/tree/master/Testing%20for%20yolov3%20tiny/YOLOv3-Series-master/od_test)

In the OD.py file, the coco.names, config and weight files were renamed. The config file is the config we used to train our model. The weight file is the trained weight which we have to mention inside the OD.py file. Inside the coco.names, we put the name of the banana as banana is the only class. 

The banana was detected when this python script was run in ubuntu but it failed to run in Raspberry Pi 4.

The screenshot below shows the section of the code where the center of the bounding box could be retrieved. 

![nnpack](images/ivan_bb.jpg)

For more understanding, please follow his [tutorial](https://www.youtube.com/watch?v=R0hipZXJjlI&list=PLZBN9cDu0MSk4IFFnTOIDihvhnHWhAa8W) .

After a lot of research,, we came across darknet-nnpack which was described in the earlier sections. Here is the link for [darknet-nnpack](https://github.com/HSRW-EMRP-WS1920-Drone/object_detection_and_landing/tree/master/Testing%20for%20yolov3%20tiny/darknet-nnpack).

With darknet-nnpack, the banana was detected after running the rpi_record.py. Like above, we changed the config and weight file names in the python script.

We needed to calculate the center of the bounding box for the detected objects. This could be retrieved from the Image.c file. The screenshot below shows the part of the script where the center of the bounding box was generated. We took this center of the bounding box for the "Landing Algorithm with object detection" code.

![nnpack](images/darknet_bb.jpg)

**A copy of our trained model with all the files except the images and CUDA files are uploaded in this Github. Here is the **[link.](https://github.com/HSRW-EMRP-WS1920-Drone/object_detection_and_landing/tree/master/darknet_Google_Colab)
The Google Colab ipython notebook describing the main steps for training the model is named as 'Tiny YOLO v3 explained.ipynb'.

### Landing algorithm with Object Detection

#### Single object detection and landing
The part with coordinates’ processing was changed to the needs of the task. Darknet library for object detection is capable to draw boxes over the detected object in the frame but doesn’t return the coordinates of the boxes for further processing. The source code of darknet (image.c file)was changed in the way that before drawing bounding box, coordinates of box are saved into a .txt file. The following lines were added:

```c
if (strcmp(names[class], "banana") ==0){ // save coordinate only for banana images
            FILE * fp; // create file link
            fp = fopen("results.txt","w"); // open / create file “results.txt” for writing
            fprintf(fp,"%.2f;%.2f",b.x*608,b.y*608); // write center coordinates of bounding box, width and height of box equals 608
            fclose(fp); // close the file
           }
```
In the landing algorithm, we need to add the center of the bounding boxes. The bounding box coordinates are read from the aforementioned .txt file and integrated to the landing algorithm as shown below:

```python
with open("results.txt","r") as file:
            for line in file:
                fields=line.split(";")
                x_cm=float(fields[0])
                y_cm=float(fields[1])
```

The z coordinate is taken from the drone’s altitude as darknet does not provide that information:

    z_cm = uav_location.alt*100.0

The first step is only changed where the drone will try to read the bounding box center of the detected object. The rest of the steps will remain the same. The code [single_banana_detection.py](https://github.com/HSRW-EMRP-WS1920-Drone/object_detection_and_landing/blob/master/scripts/single_banana_detection.py) of the modified landing algorithm is commented for better understanding. The code is not tested yet.


#### Multiple objects detection and landing

For cases of 2 and more bananas, the approach was changed to hovering over all detected objects in turn. Image.c file was changed to save to .txt file all the coordinates of the detected objects.

```c
FILE * fp;
fp = fopen("results.txt","w");
int found=-1;
for(i = 0; i < num; ++i){ //looping over all detections
…
if(class >= 0){ // if any class was found
…
 if (strcmp(names[class], "banana") ==0){
            found=1;
            fprintf(fp,"%.2f;%.2f\n",b.x*608,b.y*608);
           }
…
fclose(fp);
if (found<0){ 
 remove("results.txt")  // if nothing was detected, delete the file
}
```

Results handling in [multiple_bananas_detection.py](https://github.com/HSRW-EMRP-WS1920-Drone/object_detection_and_landing/blob/master/scripts/multiple_bananas_detection.py) also was adapted to new results.txt file content. 

All the centers of the bounding boxes of the detected objects(multiple bananas) are loaded from the text file and integrated into the landing algorithm. The idea is to hover over all the objects one by one and descend 1 meter. It can be changed with the help of dronekit library. The following codes show this process: 

```python
hover_alt_cm        =100.0
...
with open("results.txt","r") as file:
            for line in file: # for each line do coordinate processing
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
```
Hovering approach was chosen for multiple objects which is discussed in the next section of this documentation. The rest of the steps will remain same. The code [multiple_bananas_detection.py](https://github.com/HSRW-EMRP-WS1920-Drone/object_detection_and_landing/blob/master/scripts/multiple_bananas_detection.py) of the modified landing algorithm is commented for better understanding. The code is not tested yet.

#### Hovering over objects
This code is for tracking an object. It detects an object and hovers on it.
In [hovering_approach.py](https://github.com/HSRW-EMRP-WS1920-Drone/object_detection_and_landing/blob/master/scripts/hovering_approach.py) `hover_alt_cm` variable was added to define the altitude of drone.

The part of code in the first step dealing with the new location was changed 

from 
  ```python
   #-- If angle is good, descend
                    if check_angle_descend(angle_x, angle_y, angle_descend):
                        print("Low error: descending")
                        location_marker         = LocationGlobalRelative(marker_lat, marker_lon, uav_location.alt-(land_speed_cms*0.01/freq_send))
                    else:
                        location_marker         = LocationGlobalRelative(marker_lat, marker_lon, uav_location.alt)
```
to 

```python
  # angle checking was removed as the altitude doesn’t change anymore
                    location_marker         = LocationGlobalRelative(marker_lat, marker_lon, hover_alt_cm)
```
Part of code with command to land at specified height was also removed

```python
~~ #--- Command To Land
                If Z_cm <= Land_alt_cm:
                    If Vehicle.Mode == "Guided":
                        Print(" -->>Commanding To Land<<")
                        Vehicle.Mode = "Land"~~
```

```python
hover_alt_cm        =100.0  // 1 meter
...
with open("results.txt","r") as file:
            for line in file:# for each line do coordinate processing
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
```

The rest of the steps will remain the same. The code [hovering_approach.py](https://github.com/HSRW-EMRP-WS1920-Drone/object_detection_and_landing/blob/master/scripts/hovering_approach.py) of the modified landing algorithm is commented for better understanding. The code is not tested yet.