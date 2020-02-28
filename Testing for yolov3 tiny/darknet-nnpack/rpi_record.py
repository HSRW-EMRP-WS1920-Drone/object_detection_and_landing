from picamera import PiCamera
from subprocess import Popen, PIPE
import threading
from time import sleep
import os, fcntl
import cv2
from shutil import copyfile

iframe = 0

camera = PiCamera()

camera.resolution = (416, 416)

camera.capture('frame.jpg')
sleep(0.1)

yolo_proc = Popen(["./darknet",
                   "detect",
                   "./cfg/yolov3-tiny.cfg",
                   "./yolov3-tiny.weights",
                   "-thresh","0.1"],
                   stdin = PIPE, stdout = PIPE)

fcntl.fcntl(yolo_proc.stdout.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)

iframe = 0

while True:
    try:
        stdout = yolo_proc.stdout.read()
        if 'Enter Image Path' in stdout:
            try:
               im = cv2.imread('predictions.png')
               copyfile('predictions.png', 'frame%03d.png' % iframe)
               iframe += 1 
               cv2.imshow('yolov3-tiny',im)
               key = cv2.waitKey(5)
               
            except Exception:
               pass
            camera.capture('test.jpg')
            yolo_proc.stdin.write('test.jpg\n')
        if len(stdout.strip())>0:
            print('get %s' % stdout)
    except Exception:
        pass
