import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import cv2
from sklearn.utils import shuffle

#Load Images from Swing
loadedImages = []
for i in range(0, 300):
    image = cv2.imread('Dataset/Swing/swing_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#Load Images From Palm
for i in range(0, 300):
    image = cv2.imread('Dataset/Palm/palm_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))
    
#Load Images From Fist
for i in range(0, 300):
    image = cv2.imread('Dataset/Fist/fist_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#Load Images Frome Like
for i in range(0, 300):
    image = cv2.imread('Dataset/Like/like_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#Load Images From Peace
for i in range(0, 300):
    image = cv2.imread('Dataset/Peace/peace_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))


# Create OutputVector
outputVectors = []
for i in range(0, 300):
    outputVectors.append([1, 0, 0, 0, 0])

for i in range(0, 300):
    outputVectors.append([0, 1, 0, 0, 0])

for i in range(0, 300):
    outputVectors.append([0, 0, 1, 0, 0])

for i in range(0, 300):
    outputVectors.append([0, 0, 0, 1, 0])

for i in range(0, 300):
    outputVectors.append([0, 0, 0, 0, 1])


testImages = []

#Load Images for swing
for i in range(0, 300):
    image = cv2.imread('Dataset/SwingTest/swing_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

#Load Images for Palm
for i in range(0, 300):
    image = cv2.imread('Dataset/PalmTest/palm_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))
    
#Load Images for Fist
for i in range(0, 300):
    image = cv2.imread('Dataset/FistTest/fist_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

#Load Images for Like
for i in range(0, 300):
    image = cv2.imread('Dataset/LikeTest/like_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

#Load Images for Peace
for i in range(0, 300):
    image = cv2.imread('Dataset/PeaceTest/peace_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

testLabels = []

for i in range(0, 300):
    testLabels.append([1, 0, 0, 0, 0])
    
for i in range(0, 300):
    testLabels.append([0, 1, 0, 0 ,0])

for i in range(0, 300):
    testLabels.append([0, 0, 1, 0, 0])

for i in range(0, 300):
    testLabels.append([0, 0, 0, 1, 0])

for i in range(0, 300):
    testLabels.append([0, 0, 0, 0, 1])

# Define the CNN Model using Convolution Neural Network
tf.reset_default_graph()
convnet=input_data(shape=[None,89,100,1],name='input')
convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=fully_connected(convnet,1000,activation='relu')
convnet=dropout(convnet,0.75)

convnet=fully_connected(convnet,5,activation='softmax')

convnet=regression(convnet,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

model=tflearn.DNN(convnet,tensorboard_verbose=0)


# Shuffle Training Data
loadedImages, outputVectors = shuffle(loadedImages, outputVectors, random_state=0)

# Train model
model.fit(loadedImages, outputVectors, n_epoch=50,
           validation_set = (testImages, testLabels),
           snapshot_step = 100, show_metric = True, run_id = 'convnet_coursera')

model.save("TrainedModel/GestureRecogModel.tfl")