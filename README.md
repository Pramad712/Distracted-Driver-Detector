# Distracted Driver Detector
A distracted driver detector that uses deep learning and makes a "beep" sound if the driver has been distracted for at least two seconds. Made for 2023 Ardenwood Elementary School Science and Engineering Fair (6th Grade).

## Dependencies (latest as of when this project was created)
### • Python 3.10 (3.11 was too new to support most of the following libraries)
### • PyTorch: torch 1.13.1, torchvision 0.14.1 for deep learning & image augumentation
### • Pillow (PIL) 9.4.0 for more image augumentation
### • OpenCV (cv2) 4.5.5.64 for the camera (4.7.0.68 didn't work on PyCharm)
### • Pygame 2.1.3dev8 for sounds

# How it Works
First, all training and test data become bluer by blending the image with a solid blue image. Next, a ResNet-152 model is initiated that has been pre-trained on the ImageNet dataset by PyTorch. It is then trained on almost 17,000 images and tested on a tenth of that six times and saved on a file. Lastly, when the user wants to turn on the detector, OpenCV will be used to get live images from the camera and send it to the model. If the driver is distracted for at least two seconds, a “beep” sound will be played using Pygame.

# Bugs and Issues
### • No bugs have been found so far. However, PLEASE USE THIS APPLICATION IN A CAR! Look at the data shown in the folders to find a good camera placement.

# Improvements
### • Put on a small device such as a Rasberry Pi and vibrate the steering wheel/seat instead of making an annoying sound.
