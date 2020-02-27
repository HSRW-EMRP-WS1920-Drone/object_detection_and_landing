from picamera import PiCamera
from time import sleep

camera = PiCamera()
camera.resolution = (640,480)
camera.start_preview()
sleep(7)
for i in range(2):
	camera.capture("/home/pi/Documents/image"+str(i)+".jpg")
        print('Capture done= ',i)
	sleep(4)
	print('Nowwwwwwwwwwwwww')

camera.stop_preview()