# Object Detection and drone landing

By Ilgar Rasulov (Ilgar.Rasulov@hsrw.org), Md. Rakib Hassan (Md-Rakib.Hassan@hsrw.org) and Prabhat Kumar (prabhat.kumar@hsrw.org).
### (neural network for object detection) - Darknet can be used on [Linux](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux) and [Windows](https://github.com/AlexeyAB/darknet#how-to-compile-on-windows-using-vcpkg)

More details: http://pjreddie.com/darknet/yolo/

### Table of contents

1.  [Raspberry Pi setup](#raspberry-pi-setup)
    * [Operating system installation on Raspberry Pi](#operating-system-installation-on-raspberry-pi) 
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
    * [Testing phase](#testing-phase)
5.  [Troubleshooting](#troubleshooting)
6.  [Computer Vision](#computer-vision)
    * [CNN and Transfer learning](#cnn-and-transfer-learning)
    * [Object Detection](#object-detection)
    * [Yolo](#yolo)
    * [mAP](#map)
7.  [Yolo implementation](#yolo-implementation)
    * [Data Collection](#data-collection)
    * [Annotation](#annotation)
    * [GPU Support](#gpu-suppor)
    * [Training tiny yolo v3 in Google Colab](#training-tiny-yolo-v3-in-google-colab)
    * [NNPack](#nnpack)
    * [Testing the model](#testing-the-model)
8.  [Landing algorithm with Object Detection](#landing-algorithm-with-object-detection)
    * [Multiple objects detection and landing](#multiple-objects-detection-and-landing)
    * [Hovering over objects](#hovering-over-objects)
9.  [References](#references)

### Raspberry Pi setup
#### Operating system installation on Raspberry Pi
#### Remote Access to the Raspberry Pi

The execution of python program required access to command line and a keyboard for typing. Due to the fact, that during experiment companion computer (Raspberry Pi) was placed on the drone, direct connection of monitor and keyboard was impossible. The connection over network was decided as the optimal solution. 

In order to access the Raspberry Pi over network, both Raspberry and PC/laptop should be in the same net and the IP address of Raspberry Pi should be defined. First approach was to connect both devices to common WLAN. But as the university WLAN didn’t give access to connected devices’ IP addresses and third part tools were not reliable, this approach failed. Instead the second approach with WiFi hotspot was choosen.

Windows 10 provides feature of Mobile Hotspot setting:

![](images/mob_hotspot.jpg)

In this dialog network name (SSID), password and band should be set. For band options of 2.4 and 5 GHz are available. Raspberry Pi didn’t connect to 5 GHz band, so 2.4 GHz was choosen.

Network on Rpi also should be configured. The easy way to do is to use monitor, keyboard and mouse only once to connect to hotspot through standart menu:

![](images/rpiwifi.png)

https://www.circuitbasics.com/how-to-set-up-wifi-on-the-raspberry-pi-3/ 

If monitor and input devices are not accessible for initial setup, the configuration files should be added to the SD Card with raspbian image.

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

Remember DeviceID, in this case it is “\\.\PHYSICALDRIVE1”
1.4) navigate to VirtualBox folder

![](images/mob_hotspot5.jpg)

1.5) create Virtual Machine Disk file (VMDK), use as an argument device id (\\.\PHYSICALDRIVE1)

![](images/mob_hotspot6.jpg)

If the file was successfully created, message will be shown

![](images/mob_hotspot7.jpg)

1.6) Launch VirtualBox as an administrator

![](images/wificonf1.png)

1.7) Navigate to “Settint” > “Storage”.
1.8) Click on “Controller: SATA”.
1.9) Check “Use Host I/O Cache” check box.
1.10) click on “Adds hard disk” icon.

![](images/wificonf2.png)
1.11) Select “Choose Existing Disk”.

1.12) Select “C:\sdcard.vmdk” file which created in Step 1.5.

1.13) Launch the virtual machine and verify that sd card appears in the list of devices. For that run the “fdisk -l” command. In this example Linux recognized sd card as /dev/sdb

![](images/RaspberryPi_connection2.png)

VirtualBox steps are over at this point. Further steps are provided for Linux environment.

1.14) Disk should be mounted to be accessible in Linux
Type the command “mkdir /mnt/SD” to create a mount point. Change /mnt/sD with your path.
1.15) Type the command
"mount /dev/sdb1 /mnt/SD"
to mount the sdCard
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
After first boot, Linux will move this file to /etc/wpa_supplicant/ folder.

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

Connected devices IP addresses can be viewed from the table below. Currently maximal number of devices is 8.

7) Download PyTTY (https://www.putty.org/ ) for SSH connections

In the main menu enter the IP address of the Raspberry Pi, for connection type select SSH and click Open:

![](images/RaspberryPi_connection_putty.png)

8) Enter login and password for access. Default login is “pi”, and password is “raspberry”

![](images/RaspberryPi_connection.png)

#### Camera Calibration and code explanation
Any camera that you will be using, it needs to be calibrated as well. In case of using a raspberry pi on the drone, it is better to use the raspberry camera, because of its lightweight and easy connection to the Pi 

For calibrating the camera you will be required OpenCV Library, it is one of the most widely used packages for video detection, motion detection, image recognition, and deep learning.

Follow the codes one by one in order to install and compile OpenCV 4.1.2 on Raspbian OS

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
image type - type of these images, default "jpg"
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
   ```python
 #--------- Save result
    filename = workingFolder + "/cameraMatrix.txt"
    np.savetxt(filename, mtx, delimiter=',')
    filename = workingFolder + "/cameraDistortion.txt"
    np.savetxt(filename, dist, delimiter=',')

    mean_error = 0
    for i in xrange(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print("total error: ", mean_error/len(objpoints))
```
Calibration computes a set of camera indicators (camera matrix, distortion) for specific camera and should be done only once,  unless camera's optic characteristics change.

### Drone part

Select the drone as per your requirement, after selecting the type of drone-based on the motors or propellers, weather it is quadcopter or hexacopter it has to be calibrated.
It can be calibrated with the help of mission planner software
To calibrate follow the steps one by one: -
    1.  Download and install the mission planner software from here (https://firmware.ardupilot.org/Tools/MissionPlanner/).
    2. Click on INITIAL SETUP > Install Firmware > (based on your number of motors in drone select the ArduCopter version)
    3. After connecting from drone pixhawk to your laptop or pc click on CONNECT in mission planner
    4. Go to INITIAL SETUP click on Frame type (select the frame according to your drone).
    5. Followed to that in INITIAL SETUP calibrate: -
    i. Accelerometer Calibration
    ii. Compass Calibration
    iii. Radio Calibration

i. Accelerometer Calibration:

![](images/drone_1.png)

https://ardupilot.org/copter/docs/common-accelerometer-calibration.html

Click Accel Calibration to start the calibration.
Mission Planner will prompt you to place the vehicle in each calibration position. Press any key to indicate that the autopilot is in position and then proceed to the next orientation.
The calibration positions are: level, on the right side, left side, nose down, nose up and on its back.

ii. Compass Calibration: -

![](images/drone_2.png)

https://ardupilot.org/copter/docs/common-compass-calibration-in-mission-planner.html

To perform the onboard calibration:
    • Click the “Onboard Mag Calibration” section’s “Start” button
    • if your autopilot has a buzzer attached you should hear a single tone followed by short beep once per second
    • hold the vehicle in the air and rotate it so that each side (front, back, left, right, top and bottom) points down towards the earth for a few seconds in turn. Consider a full 360-degree turn with each turn pointing a different direction of the vehicle to the ground. It will result in 6 full turns plus possibly some additional time and turns to confirm the calibration or retry if it initially does not pass. 
    • As the vehicle is rotated the green bars should extend further and further to the right until the calibration completes
    • upon successful completion, three rising tones will be emitted and a “Please reboot the autopilot” window will appear and you will need to reboot the autopilot before it is possible to arm the vehicle.

If calibration fails:
    • you will hear an “unhappy” failure tone, the green bars may reset to the left, and the calibration routine may restart (depending upon the ground station). Mission Planner will automatically retry, so continue to rotate the vehicle as instructed above.
    • if a compass is not calibrating, consider moving to a different area away from magnetic disturbances, and remove electronics from your pockets.
    • if, after multiple attempts, the compass has not passed the calibration, press the “Cancel” button and change the “Fitness” drop-down to a more relaxed setting and try again.
    • if compass calibration still fails it may help to raise COMPASS_OFFS_MAX from 850 to 2000 or even 3000
    • finally, if a single compass is not calibrating and you trust the others, disable it
    

iii. Radio Calibration

![](images/drone_3.jpg)

![](images/drone_4.jpg)

https://ardupilot.org/copter/docs/common-radio-control-calibration.html

It is the calibration between controller and drone, to set the proper connection of the transmitter and the receiver in both the devices.
Calibration
    • Click on the green “Radio Calibration” button on the bottom right
    • Press “OK”, when prompted to check the radio control equipment, is on, the battery is not connected, and propellers are not attached

Move the transmitter’s control sticks, knobs, and switches to their limits. Red lines will appear across the calibration bars to show the minimum and maximum values seen so far
    • Select Click when Done
    • A window will appear with the prompt, “Ensure all your sticks are centered and the throttle is down and click ok to continue”. Move the throttle to zero and press “OK”.
    • Mission Planner will show a summary of the calibration data. Normal values are around 1100 for minimums and 1900 for maximums.

### Raspberry Pi to Pixhawk connection
#### Hardware Communication
Communication is done through Telem2 telemetry port of Pixhawk. Ground, Rx (receiver), Tx(transmitter) and +5V should be connected to 4 pins of Raspberry Pi.

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

In the case of successful connection the output should be the follwing

![](images/RaspberryPi_ArmTestThroughPutty.png)

### Landing with aruco marker approach
#### Aruco marker

An ArUco marker is a black square marker with inner binary representation of identifier. Having black borders, these markers are easy to detect in a frame.

![aruco](images/aruco_markers.jpg)
> Aruco Marker Samples

For each the specific application a dictionary – a set of markers – is defined. Dictionaries have such properties as the dictionary size and the marker size. The size of dictionary is defined by the number of markers it contains, and the marker size is the number of bits it has in the inner part.
The identification code of marker is not the result of conversion of binary image to a decimal base, but the marker index in the dictionary. The reason is that for high number of bits the results may become unmanagable. 

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
-      the size of marker in pixels
-      output image
-      black border width, expressed by the number of internal bits (optional, default equals 1)

There are also online aruco generators, one of which was used in this project. As an example, http://chev.me/arucogen/ website can be mentioned.

#### Marker detection

The goal of marker detection is to return the position and id of the each marker found in the image. This process can be divided to two steps:

1.  Possible candidates’ detection – return all the square shapes and discard non convex ones, by analyzing contours of each figure.
2.  For each candidate its inner codification is analyzed. Several actions are taken as:
    • transformation to canonical form
    • black and white bits are separated
    • division to cells in accordance with marker size
    • the number of black and white pixels is counted to determine the color of cell
    • the bits are analyzed to their relevance to the selected dictionary

Results of detection can be visualised

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

Camera pose can be obtained from the markers position if the camera was calibrated (camera matrix and distortion coefficients are available). The camera pose is the transformation from the marker coordinate system to the camera coordinate system. Rotation and transformation are estimated in the form of vectors.

```c
cv::Mat cameraMatrix, distCoeffs;

std::vector<cv::Vec3d> rvecs, tvecs;
cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);
```
There rvecs and tvecs are rotation and transformation vehicles respectively.

#### Landing algorithm and code explanation

The algorithm steps:

1) check the aruco class object for new detection. ArucoSingleTracker  is a wrapper class for the access to aruco API.

```c
marker_found, x_cm, y_cm, z_cm = aruco_tracker.track(loop=False) 
```

loop = False – do detection once.

Output:
marker_found – flag, indicating that marker was found, boolean
x_cm – x coordinate of the marker on the image
y_cm – y coordinate of the marker on the image
z_cm – z coordinate of the marker on the image, for the images taken from less than 5 meters, z_cm is taken as an altitude of the drone.

2) If marker_found is True convert x and y coordinates from camera coordinate system to drone coordinate system. The formula is

![aruco](images/coords_to.png)
![aruco](images/coords_to2.png)

where ![aruco](formulas/f1.jpg) are board frame coordinates, ![aruco](formulas/f2.jpg) are camera frame coordinates.

3) Drone can navigate to the location of marker with or without altitude reduction. The decision is done according the angle between drones vertical axis and the vector to the marker. This angle indicates drones closeness to the marker and commands to move with landing when it is less than some threshold values. The command  of moving with descend is given if the expression

 ![aruco](formulas/f3.jpg)
 
returns true value, where  ![aruco](formulas/f4.jpg) and  ![aruco](formulas/f5.jpg) in radians.

4) Calculate latitude and longitude from the marker coordinates. The algorithm was taken from gis portal and is relatively accurate over small distances (10m within 1km). First, north and east attitudes should be calculated for the current yaw of drone. Yaw is a rotation indicator in horizontal space.

 ![aruco](images/Yaw_Axis_Corrected.svg.png)

 ![aruco](formulas/f6.jpg)
 
  ![aruco](formulas/f7.jpg)

Second, latitude and longitude are calculated. Drone’s coordinate is taken from GPS, earth radius is taken approximately 6378137 meters.

![aruco](formulas/f8.jpg)

![aruco](formulas/f9.jpg)

![aruco](formulas/f10.jpg)

![aruco](formulas/f11.jpg)

5) If the height of drone is less than some threshold altitude, perform vertical landing by changing the mode of vehicle to “LAND” value.

```c
vehicle.mode = "LAND" 
```
#### Testing phase
### Troubleshooting
#### Troubleshooting: Raspberry Pi Camera
Sometimes camera doesnot get detected so use the following commands to solve the problem:
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

You can run the python script at bootap with Crontab. We were running the python script in the testing phase with remote access. It is better to enable automatic execution of python script at specific time of the day or when the Raspberry Pi boots up. First we need to run the code:

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
    # │ │ │ │ └───── day of week (0 - 7) (0 to 6 are Sunday to Saturday, or use names; 7 is Sunday, the same as 0)
    # │ │ │ └────────── month (1 - 12)
    # │ │ └─────────────── day of month (1 - 31)
    # │ └──────────────────── hour (0 - 23)
    # └───────────────────────── min (0 - 59)


To run a command every time the Raspberry Pi starts up, write @reboot instead of the time and date. For example:

```c
@reboot python /home/pi/video_with_timestamp.py
```

Source link to read in detail: https://www.raspberrypi.org/documentation/linux/usage/cron.md
https://crontab-generator.org/

For example:
If you want to run the code every 10 Minutes, between 8AM and 8PM, from Monday through Friday or  have a task that should only run during normal business hours, then this could be accomplished using ranges that for the hour and weekday values separated by a dash.

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
#### CNN and Transfer learning
#### Object Detection
#### Yolo
#### mAP

### Yolo implementation
#### Data Collection
#### Annotation
#### GPU Support
#### Training tiny yolo v3 in Google Colab
In order to train our banana model we used open source neural network framework darknet (https://pjreddie.com/darknet/). (AlexyAB github instructions)

Benefits:
- Fast and Easy to install
- CPU and GPU computation support

##### STEP 1. Connect the Colab notebook to Google Drive
One advantage of using Google Colab is that the data can be accessed directly from the google drive. We have to mount the drive on the Colab,then Colab will send the link for autorization code and offer input for this code typing. If drive was already mounted, to force remount of google drive, the following parameter should be added to mount method
force_remount=True

```python
drive.mount('/content/gdrive', force_remount=True)
```
 
```python
# These commands import the drive library and mounts your Google Drive as a local drive. You can access to your Drive files using this path "/content/gdrive/My Drive/"
 
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

Version of cuDNN for linux (archived as tgz) was downloaded from https://developer.nvidia.com/ and uploaded to google drive.


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

They should be put into anchors field of each [yolo] block of cfg file.

![darknet](images/darknet_conf3.jpg)

##### Initial weights file preparation
Weights file is required for training model in darknet. This enables *continuing* training from the point where algorithm stopped, as darknet saves weights file after every 1000 iterations. But for the first run of command there is not yet any file generated. In this case standart weights file "yolov3-tiny.weights" may be used. Another possible solution is to generate initial weights file by running command

`!./darknet partial yolov3-tiny0802.cfg yolov3-tiny.weights weights/yolov3-tiny.conv.15 15`

which will save file to "yolov3-tiny.conv.15" in this case.

##### Training model

In order to train model the following command should be run in Linux (Google Colab implementation)

`!./darknet detector train <data> <config> <weights> -dont_show -map`

, where 
- data – is a path to obj.data file (path to the image and backup folder),
- config is a path to configuration file(hyperparameter tuning),
- weights is a path to initial weights file
- -dont_show flag tells the program to suppress display output
- -map tells to generate mAp (mean average precision) diagram

`!./darknet detector train data/obj.data yolov3-tiny0802.cfg weights/yolov3-tiny.conv.15 -dont_show -mjpeg_port 8090 -map`

##### When should I stop training

Usually one class requires about 2000 iterations, and 4000 iterations is a minimum for all classes.
Darknet outputs average error on eacht iteration. Stop training when no longer avg loss decreases.

When training is over, different weight files (for each thousand iteration count) in /darknet/backup folder can be used in detection and the best one is choosed. Using weights file with maximum iterations is not always the best solution for the **Overfitting** phenomena when model is trained too much and can recognize only the objects, which look exactly like the training set images.

##### How to calculate mAP
Mean Average Precision (mAP) chart can be calculated while model is learning. To do this in training command *-map* parameter should be added:

`darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -map`

##### How to improve object detection:

in .cfg-file to increase precision

- set flag random=1

- increase network resolution (height=608, width=608 or any value multiple of 32)

Also 

- check that each object is labeled in dataset

- run training with `-show_imgs` argument to see bounding boxes

- include into dataset images with many other objects and images without target object

##### Mean Average Precision for our model

![map](images/map.png)

##### Tiny YOLOv3 results for banana


#### NNPack
Despite the fact that YoloV3 is effecient, it still requires computational power from the device it operate on. Simplified version of Yolo - Yolov3-tiny - runs faster but detects less accurately. Both these approaches didn't find object when tested on Raspberry Pi.
Decision was made to use NNPACK - a library for the neural network to run on multi-core CPU. The following steps should be completed to use NNPack.

**Step 0**
Prepare Python and camera.

Install  package installer for Python *pip*

`sudo apt-get install python-pip`

Enable camera in Raspberry Pi. For it run code:

`sudo raspi-config`

Navigate to `Interfacing Options`, and enable `Camera`
To be able to use camera , reboot is required.

**Step 1**
**NNPack installation**

NNPack is used to optimize Darknet for embedded devices with ARM CPUs without using a GPU. It uses transform-based convolution computation which allows 40% faster performance on non-initial frames. Particulary, for repeated inferences, ie. video, NNPack is more beneficial to use.

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

### Landing algorithm with Object Detection

The part with coordinates’ processing was changed to the needs of the task. Darknet library for object detection is capable to draw boxes over the detected object in the frame, but doesn’t return the coordinates of the boxes for further processing. The source code of darknet (image.c file)was changed in the way that before drawing bounding box, coordinates of box are saved into a .txt file. The following lines were added:

```c
if (strcmp(names[class], "banana") ==0){ // save coordinate only for banana images
            FILE * fp; // create file link
            fp = fopen("results.txt","w"); // open / create file “results.txt” for writing
            fprintf(fp,"%.2f;%.2f",b.x*608,b.y*608); // write center coordinates of bounding box, width and height of box equals 608
            fclose(fp); // close the file
           }
```

in python file coordinates are read from that file:

```python
with open("results.txt","r") as file:
            for line in file:
                fields=line.split(";")
                x_cm=float(fields[0])
                y_cm=float(fields[1])
```

The z coordinate is taken from the drone’s altitude as darknet does not provide that information:

    z_cm = uav_location.alt*100.0

### Multiple objects detection and landing

For cases of 2 and more bananas the approach was changed to hovering over all detected objects in turn. Image.c file was changed to save to .txt file all the coordinates

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

Results handling in Python also was adapted to new results.txt file content. hover_alt variable was added to define the altitude of drone’s hovering.

```c
hover_alt_cm        =100.0  // 1 meter
...
with open("results.txt","r") as file:
            for line in file: 
   # for each line do coordinate processing
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
                        #-- If angle is good, descend
   # angle checking was removed as altitude doesn’t change anymore
                    location_marker         = LocationGlobalRelative(marker_lat, marker_lon, hover_alt_cm)

                    vehicle.simple_goto(location_marker)
                    print("UAV Location    Lat = %.7f  Lon = %.7f",uav_location.lat, uav_location.lon)
                    print("Commanding to   Lat = %.7f  Lon = %.7f",location_marker.lat, location_marker.lon)
```
### References