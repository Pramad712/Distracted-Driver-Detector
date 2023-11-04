# Distracted Driver Detector
A distracted driver detector that uses deep learning and makes a "beep" sound if the driver has been distracted for at least two seconds. 

## Dependencies (latest as of when this project was created)
### • Python 3.10 (3.11 was too new to support most of the following libraries)
### • PyTorch: torch 1.13.1, torchvision 0.14.1 for deep learning & image augumentation
### • Pillow (PIL) 9.4.0 for more image augumentation
### • OpenCV (cv2) 4.5.5.64 for the camera (4.7.0.68 didn't work on PyCharm)
### • Pygame 2.1 for sounds

# Warnings, Instructions and Notes
### • *Only use this application only inside a car*. Look at the images in the "data.zip" folder to find a good camera placement.
### • Only run the file "main.py". If you run "deep_learning_model.py", then it will take hours to train the model.
### • Only 750 images were put in the "data" folder. However, over 17,000 images were used for training & testing. If you want the entire dataset, see [this](https://www.kaggle.com/datasets/rightway11/state-farm-distracted-driver-detection). Look at the "train" folder under the "imgs" folder only. Credits to Kaggle and State Farm for making the dataset.

# How it Works
First, all training and test data become bluer by blending the image with a solid blue image. Next, a ResNet-152 model is initiated that has been pre-trained on the ImageNet dataset by PyTorch. It is then trained on almost 17,000 images and tested on a tenth of that six times and saved on a file. Lastly, when the user wants to turn on the detector, OpenCV will be used to get live images from the camera and send it to the model. If the driver is distracted for at least two seconds, a “beep” sound will be played using Pygame.

Even though there are only two possible outcomes per image (distracted and not distracted), it may cause a lot of false-positives and false-negatives. Hence, the test data is split into ten catagories: not distracted, holding a phone with the right hand, holding a phone with the head on the right, holding a phone with the left hand, holding a phone with the head on the left, reaching for something, drinking something, reaching for something behind on the right, focusing on the face, and facing right.

# Graphs
![science_fair_2023_big_confusion_matrix](https://user-images.githubusercontent.com/77818951/218295213-1af42226-f30f-4294-88a3-c6cbe9dfe8cc.png)
![science_fair_2023_small_confusion_matrix](https://user-images.githubusercontent.com/77818951/218295216-9f2e60ce-1e28-455d-a9c0-8ad0a17ee661.png)

## Interpretation
The graphs shown are both confusion matrices, which display the accuracy of a model whenever it predicts a certain class. The columns are the predicted class, and the rows are the actual class. In the first graph, the classes are ordered in the same ordering as mentioned above. All values are sub-75, except for the correct predictions, which are over 1,200, which shows significant accuracy. Since the second one has just two rows and columns for distracted and not distracted, more statistics is added. The bottom-left most number is the sensitivity, or the number of true positives divided by the number of predicted positives, and the number to the left of it is the specificity, or the number of true negatives divided by the number of predicted negatives. The top-right most number is precision, the number of true-positives divided by the number of actual positives, and the one below it is the negative predictive value, or the number of true-negatives divided by the number of actual negatives. Lastly, the bottom-most number is the accuracy - 98%. All these statistics are 98% or higher, except for the specificity and negative predictive value, which are still about 90% and 93% respectively.

# Sample Predictions
Note: The images are screeshots, so the borders are off.

![image](https://user-images.githubusercontent.com/77818951/219971544-42e65807-b08a-4e4d-9a14-ff7debe00e70.png)
![image](https://user-images.githubusercontent.com/77818951/219971789-4057c7f1-0c78-42df-871c-f285314d0e56.png)
![image](https://user-images.githubusercontent.com/77818951/219971579-f6f7205a-3e40-4332-b3cf-4d26b406f95f.png)
![image](https://user-images.githubusercontent.com/77818951/219971639-4814a824-a591-4fb6-9f7d-189b0d6da018.png)
![image](https://user-images.githubusercontent.com/77818951/219971668-3031e34f-24f4-4581-ba49-198f9238fcd3.png)
![image](https://user-images.githubusercontent.com/77818951/219971762-f6c94e9c-13e1-4901-987d-59899693731a.png)

# Improvements
### • Put the code on a small device such as a Rasberry Pi and vibrate the steering wheel/seat instead of making a sound.
