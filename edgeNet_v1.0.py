'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import img_to_array
import glob
import cv2
import random
batch_size = 128
num_classes = 10
epochs = 100


def addNoise (img):
    #img = ...    # numpy-array of shape (N, M); dtype=np.uint8
    # ...
    mean = 0.0   # some constant
    std = 1.0    # some constant (standard deviation)
    noisy_img = img + np.random.normal(mean, std, img.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255)
    return noisy_img_clipped
def rotateImage(image, angle):
    #print (image.shape)
    row,col = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image


def randomCrop(img):
	#print (img.shape)
	y,x= img.shape
	#print (y,x)
	crop_img = img[0:y-random.randint(0,5), 0:x-random.randint(0,5)]
	#print (crop_img.shape)
	crop_img = cv2.resize(crop_img, (28, 28))
	#cv2.imwrite("./GT.png",img)
	#cv2.imwrite("./tempCrop.png",crop_img)

	return crop_img

def prosTransformation(img):
    rows,cols= img.shape

    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(img,M,(300,300))
    return dst

def translation(img):
    rows,cols = img.shape

    M = np.float32([[1,0,random.randint(-5,5)],[0,1,random.randint(-5,5)]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst


trainDir = "./unifiedDataset/Train/"
testDir = "./unifiedDataset/Test/"
trainImageList = glob.glob(trainDir+"*/*.png")
print (len(trainImageList), trainImageList[0].split("/")[-2])
testImageList =  glob.glob(testDir+"*/*.png")
print (len(testImageList), testImageList[0].split("/")[-2])
random.seed(42)
random.shuffle(trainImageList)
random.shuffle(testImageList)
img_rows, img_cols = 28 , 28
input_shape = (img_rows, img_cols, 1)


trainImage =[]
edgeTrainImage = []
trainLabel = []
testImage =[]
edgeTestImage = []
testLabel = []
i=0
for imagePath in trainImageList:
    #Read Input Training Sample (1 Channel)
    inputImage = cv2.imread(imagePath,0)
    inputImage = 255-inputImage

    #Make All Images black and white
    #inputImage = invImg
    #Process Input Image for Training
    trainImage.append(inputImage)
    cv2.imwrite("./inputImage.png",inputImage)
    #invImg = 255-inputImage
    edgeImage = cv2.Canny(inputImage,100,200)
    cv2.imwrite("./edgeImage.png",edgeImage)
    edgeImage = edgeImage.reshape( 28, 28,1 )
    edgeTrainImage.append(edgeImage)
    trainLabel.append(imagePath.split("/")[-2])

    #Augment Input Image with Random Rotation
    rotAngel = random.randint(-45,45)
    imgRot = rotateImage(inputImage,rotAngel)
    trainImage.append(imgRot)
    #invImg = 255-imgRot
    edgeImage = cv2.Canny(imgRot,100,200)
    #edgeImage = 255 - edgeImage
    #cv2.imwrite("./edgeImage.png",edgeImage)
    edgeImage = edgeImage.reshape( 28, 28,1 )
    edgeTrainImage.append(edgeImage)
    trainLabel.append(imagePath.split("/")[-2])

    #Augment Input Image with Block Effect
    blocEffect = cv2.resize(inputImage, (14, 44))
    blocEffect = cv2.resize(blocEffect, (28, 28))
    trainImage.append(blocEffect)
    #invImg = 255-blocEffect
    edgeImage = cv2.Canny(blocEffect,100,200)
    edgeImage = edgeImage.reshape( 28, 28,1 )
    edgeTrainImage.append(edgeImage)
    trainLabel.append(imagePath.split("/")[-2])

    #Augment Input Image with Translation
    transImage = translation(inputImage)
    trainImage.append(transImage)
    #invImg = 255-transImage
    edgeImage = cv2.Canny(transImage,100,200)
    edgeImage = edgeImage.reshape( 28, 28,1 )
    edgeTrainImage.append(edgeImage)
    trainLabel.append(imagePath.split("/")[-2])

	

# convert class vectors to binary class matrices
trainImage = np.array(trainImage, dtype="float") / 255.0
trainLabel = np.array(trainLabel)
trainImage = trainImage.reshape( trainImage.shape[0],  28, 28,1 )
edgeTrainImage = np.array(edgeTrainImage, dtype="float") / 255.0#np.asarray(edgeTrainImage).astype('float32')
edgeTrainImage = edgeTrainImage.reshape( edgeTrainImage.shape[0], 28, 28 , 1)
print ("checking shape of the input array(s)",trainImage.shape, edgeTrainImage.shape)

i=0
for imagePath in testImageList:
    #Read Input Testing Sample (1 Channel)
    inputImage = cv2.imread(imagePath,0)
    inputImage = 255-inputImage
    #Process Input Image for Testing
    testImage.append(inputImage)

    edgeImage = cv2.Canny(inputImage,100,200)
    edgeImage = edgeImage.reshape( 28, 28,1 )
    edgeTestImage.append(edgeImage)
    testLabel.append(imagePath.split("/")[-2])

# convert class vectors to binary class matrices
testImage = np.array(testImage, dtype="float") / 255.0
testLabel = np.array(testLabel)
testImage = testImage.reshape( testImage.shape[0],  28, 28,1 )
edgeTestImage = np.array(edgeTestImage, dtype="float") / 255.0#np.asarray(edgeTrainImage).astype('float32')
edgeTestImage = edgeTestImage.reshape( edgeTestImage.shape[0], 28, 28 , 1)
print ("checking shape of the input array(s)",testImage.shape, edgeTestImage.shape)



y_train = keras.utils.to_categorical(trainLabel, num_classes)
y_test = keras.utils.to_categorical(testLabel, num_classes)

#Input PlaceHolder
inpX = Input(input_shape)
inpY = Input(input_shape)
#Input Image Layer
inputImg = Conv2D(16, kernel_size=3,strides=1,padding='same',activation='relu', name="inputImgConv")(inpX)
#inputImg = Dropout(0.25,name="inputImgDropout")(inputImg)
#Input Edge Layer
inputEdge= Conv2D(16, kernel_size=3, strides=1,padding='same',activation='relu',name="inputEdgeConv")(inpY)
#inputEdge = Dropout(0.25,name="inputEdgeDropout")(inputEdge)


#Concate Inputs 
inputMerge = Add()([inputImg, inputEdge])
#Convolutional Layer 1
layer1 = Conv2D(32, kernel_size=3, strides=1,padding='same',activation='relu',name="convLayer1")(inputMerge)
layer1 = Dropout(0.20,name="convLayer1Dropout")(layer1)
#Convolutional Layer 2
layer2 = Conv2D(32, kernel_size=3, strides=1,padding='same',dilation_rate=(2, 2),activation='relu',name="convLayer2")(layer1)
#layer2 = Dropout(0.25,name="convLayer2Dropout")(layer2)
#Convolutional Layer 3
layer3 = Conv2D(32, kernel_size=3, strides=1,padding='same',dilation_rate=(2, 2),activation='relu',name="convLayer3")(layer2)
layer3 = Dropout(0.25,name="convLayer3Dropout")(layer3)


#Edge Connection
shortCut = Add(name="edgeShortcut")([inpY,layer3]) 
#Convolutional Layer 4
layer4 = Conv2D(32, kernel_size=3, strides=1,padding='same',activation='relu',name="convLayer4")(shortCut)
layer4 = Dropout(0.25,name="convLayer4Dropout")(layer4)
#Global Pooling (Average)
layer4 = AveragePooling2D((2, 2), name="globalPoolLayer")(layer4)
#layer3 = MaxPooling2D((2, 2), name="globalPoolLayer")(layer3)


#Flatten to feed Fully-Connected L ayer
denseLayer = Flatten()(layer4)
#Fully Connected Layer 
denseLayer = Dense(128, activation='relu',name="denseLayer")(denseLayer)
denseLayer = Dropout(0.25,name="denseLayerDropout")(denseLayer)
#Softmax Classifier
outputLayer = Dense(10, activation='softmax',name="classifier")(denseLayer)

#Keras Model 
model = Model([inpX, inpY], outputLayer)

#save weights
checkpoint = ModelCheckpoint('./output/{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

#Model Summary
model.summary()
#Compile Keras Model 
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#Feed edgeNet
model.fit([trainImage,edgeTrainImage], y_train,
          batch_size=batch_size,shuffle=True,
          epochs=epochs,
		  callbacks=[checkpoint],
          verbose=2,
          validation_data=([testImage,edgeTestImage], y_test))
score = model.evaluate([testImage,edgeTestImage], y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
