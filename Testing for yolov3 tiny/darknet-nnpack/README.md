This repository is forked from https://github.com/digitalbrain79/darknet-nnpack/

A few changes had been made to make it work right out of box on Raspberry Pi 3 B
1. Minor improvement to Readme to make it easier to follow
2. Modification to Makefile to work with RPI 3
3. Added two Python files to wrap the darknet
4. Come with yolov3-tiny weight

# Step 0: prepare Python and Pi Camera
Log in to Raspberry Pi using SSH or directly in terminal.<br/>

Make sure ```pip-install``` is included (it should come together with Debian
```
sudo apt-get install python-pip
```
Install OpenCV. The simplest way on RPI is as follows (do not build from source!):
```
sudo apt-get install python-opencv
```

Enable pi camera
```
sudo raspi-config
```
Go to ```Interfacing Options```, and enable ```P1/Camera```

You will have to reboot the pi to be able to use the camera

A few additional words here. In the advanced option of raspi-config, you can adjust the memory split between CPU and GPU. Althoug we would like to allocate more ram to CPU so that the pi can load a larger model, you will want to allocate at least 64MB to GPU as the camera module would require it.

# Step 1: Install NNPACK

NNPACK was used to optimize [Darknet](https://github.com/pjreddie/darknet) without using a GPU. It is useful for embedded devices using ARM CPUs.

Idein's [qmkl](https://github.com/Idein/qmkl) is also used to accelerate the SGEMM using the GPU. This is slower than NNPACK on NEON-capable devices, and primarily useful for ARM CPUs without NEON.

The NNPACK implementation in Darknet was improved to use transform-based convolution computation, allowing for 40%+ faster inference performance on non-initial frames. This is most useful for repeated inferences, ie. video, or if Darknet is left open to continue processing input instead of allowed to terminate after processing input.

## Install Ninja (building tool)

Install [PeachPy](https://github.com/Maratyszcza/PeachPy) and [confu](https://github.com/Maratyszcza/confu)
```
sudo pip install --upgrade git+https://github.com/Maratyszcza/PeachPy
sudo pip install --upgrade git+https://github.com/Maratyszcza/confu
```
Install [Ninja](https://ninja-build.org/)
```
git clone https://github.com/ninja-build/ninja.git
cd ninja
git checkout release
./configure.py --bootstrap
export NINJA_PATH=$PWD
cd
```

~~Install clang (I'm not sure why we need this, NNPACK doesn't use it unless you specifically target it).~~
~~sudo apt-get install clang~~

## Install NNPACK

Install modified [NNPACK](https://github.com/shizukachan/NNPACK)
```
git clone https://github.com/shizukachan/NNPACK
cd NNPACK
confu setup
python ./configure.py --backend auto
```
If you are compiling for the Pi Zero, change the last line to `python ./configure.py --backend scalar`

You can skip the following several lines from the original darknet-nnpack repos. I found them not very necessary (or maybe I missed something)
~~It's also recommended to examine and edit https://github.com/digitalbrain79/NNPACK-darknet/blob/master/src/init.c#L215 to match your CPU architecture if you're on ARM, as the cache size detection code only works on x86.~~

~~Since none of the ARM CPUs have a L3, it's [recommended](https://github.com/Maratyszcza/NNPACK/issues/33) to set L3 = L2 and set inclusive=false. This should lead to the L2 size being set equal to the L3 size.~~

~~Ironically, after some trial and error, I've found that setting L3 to an arbitrary 2MB seems to work pretty well.~~

Build NNPACK with ninja (this might take * *quie* * a while, be patient. In fact my Pi crashed in the first time. Just reboot and run again):
```
$NINJA_PATH/ninja
```
do a ```ls``` and you should be able to find folder ```lib``` and ```include``` if all went well:
```
ls
```
Test if NNPACK is working:
```
bin/convolution-inference-smoketest
```
In my case, the test actually failed in the first time. But I just ran the test again and all items are passed. So if your test failed, don't panic, try one more time.

Copy the libraries and header files to the system environment:
```
sudo cp -a lib/* /usr/lib/
sudo cp include/nnpack.h /usr/include/
sudo cp deps/pthreadpool/include/pthreadpool.h /usr/include/
```

~~If the convolution-inference-smoketest fails, you've probably hit a compiler bug and will have to change to Clang or an older version of GCC.~~

~~You can skip the qmkl/qasm/qbin2hex steps if you aren't targeting the QPU.~~

~~Install [qmkl](https://github.com/Idein/qmkl)~~
~~sudo apt-get install cmake~~
~~git clone https://github.com/Idein/qmkl.git~~
~~cd qmkl~~
~~cmake .~~
~~make~~
~~sudo make install~~

~~Install [qasm2](https://github.com/Terminus-IMRC/qpu-assembler2)~~
~~sudo apt-get install flex~~
~~git clone https://github.com/Terminus-IMRC/qpu-assembler2~~
~~cd qpu-assembler2~~
~~make~~
~~sudo make install~~

~~Install [qbin2hex](https://github.com/Terminus-IMRC/qpu-bin-to-hex)~~
~~git clone https://github.com/Terminus-IMRC/qpu-bin-to-hex~~
~~cd qpu-bin-to-hex~~
~~make~~
~~sudo make install~~

# Step 2. Install darknet-nnpack
We have finally finished configuring everything needed. Now simply clone this repository. Note that we are cloning the **yolov3** branch. It comes with the python wrapper I wrote, correct makefile, and yolov3 weight:
```
cd
git clone -b yolov3 https://github.com/zxzhaixiang/darknet-nnpack
cd darknet-nnpack
git checkout yolov3
make
```

At this point, you can build darknet-nnpack using `make`. Be sure to edit the Makefile before compiling.

# Step 3. Test with Yolov3-tiny
Despite doing so many pre-configurations, Raspberry Pi is not powerful enough to run the full YoloV3 version. The YoloV3-tiny version, however, can be run at about 1 frame per second rate

I wrote two python wrappers, ```rpi_video.py``` and ```rpi_record.py```. What these two python codes do is to take pictures with PiCamera python library, and spawn darknet executable to conduct detection taks to the picture, and then save to prediction.png, and the python code will load prediction.png and display it on the screen via opencv. Therefore, all the detection jobs are done by darknet, and python simply provides in and out. ```rpi_video.py``` will only display the real-time object detection result on the screen as an animation (about 1 frame every 1-1.5 second); ```rpi_record.py``` will also save each frame for your own record (like making a git animation afterwards)

To test it, simply run
```
sudo python rpi_video.py
```
or
```
sudo python rpi_record.py
```

You can adjust the task type (detection/classification?), weight, configure file, and threshold in line
```
yolo_proc = Popen(["./darknet",
                   "detect",
                   "./cfg/yolov3-tiny.cfg",
                   "./yolov3-tiny.weights",
                   "-thresh", "0.1"],
                   stdin = PIPE, stdout = PIPE)
```

For more details/weights/configuration/different ways to call darknet, refer to the official [YOLO homepage](https://pjreddie.com/darknet/yolo/).


==================================================================================================================

You can find more information in Shizukachan's darknet-nnpack repos. He has more detailed benchmark test using different hardware platform and compiling options. 
