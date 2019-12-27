import os
import sys
from tqdm import tqdm
import numpy
import cv2

sys.path.append(os.path.abspath('../lib'))
import loadData
import buildModel
import utils

############################################################
# LOAD THE LABELLED DATASET AND SPLIT INTO TRAINING AND TEST
############################################################
xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd = loadData.loadPillarData(fileName='../dataset/newLabelledDataset.dat',numClasses=2,row=128,col=128,rotFlag=True,flipFlag=True,RGB=True)
# xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd = loadData.loadPillarData(fileName='../dataset/labelledDataset.dat',numClasses=2,row=224,col=224,rotFlag=False,flipFlag=False,RGB=True)
print (xTrain.shape,yTrainInd.shape,xTest.shape,yTestInd.shape)
# Shape of training set - 48999, 32, 32, 1
# Shape of test set - 5445, 32, 32, 1
############################################################

############################################################
# TRAIN YOUR MODEL. BUILD MODEL IN A SEPARATE FUNCTION FILE
############################################################
# buildModel.model_01(name='model_01_PBS',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=10,batchSize=1000)
# buildModel.model_02(name='model_02_newData',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=500,batchSize=1000)
# buildModel.model_03(name='model_03_newData',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=200,batchSize=1000)
# buildModel.model_04(name='model_04_newData',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=50,batchSize=200)
# buildModel.model_05(name='model_05_newData',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=50,batchSize=200)
buildModel.trainUsingVGG16(name='vgg16_newData',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=20,batchSize=200)
############################################################


############################################################
# TRAIN INTERMEDIATE MODEL WITH A DIFFERENT LEARNING RATE
############################################################
# buildModel.trainIntermediateModel(name='',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=500,batchSize=1000,learningRate=0.001)


############################################################
# RUN THE MODEL AND ON TEST AND TRAIN DATASET AND GENERATE
# IMAGES FOR FALSE IDENTIFICATION
############################################################
# modelPath = '/home/utkarsh/Projects/PillarClassification/model/model_01_intermediate_487_accuracy_trainAcc_99.94_testAcc_99.85.h5'
# trainImagesPath = '/home/utkarsh/Projects/PillarClassification/dataset/incorrectClassifications/train'
# testImagesPath = '/home/utkarsh/Projects/PillarClassification/dataset/incorrectClassifications/test'

# utils.falseClassificationImage(modelPath,trainImagesPath,xTrain,yTrain)
# utils.falseClassificationImage(modelPath,testImagesPath,xTest,yTest)
############################################################
