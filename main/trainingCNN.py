'''
Source code for training different neural network models. The code is
split into 5 sections.

Part 1 (Training)
-----------------
Section 1 : Loading the labelled dataset into system memory
Section 2 : Training the newural network based on the model definition
    in ../lib/buildModel.py. Users can select the number of epochs and
    batchsize.
Section 3 : Train an intermediate model using different optimizer with
    different learning rate.
    
Part 2 (Check performance of trained model)
-------------------------------------------
Section 4 : Identify the incorrect classified nanopillars for different
    models and save their images in the user defined directories.
Section 5 : Print the accuracy of models for the entire labelled and
    augmented dataset.
'''

import os
import sys
import numpy
import cv2
from tqdm import tqdm
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../lib'))
import buildModel
import loadData
import utils
import transform

############################################################
# SECTION 1
# LOAD THE LABELLED DATASET AND SPLIT INTO TRAINING AND TEST
############################################################
xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd = loadData.loadPillarData(fileName='../dataset/labelledDataset.dat',numClasses=2,row=32,col=32,rotFlag=True,flipFlag=True,RGB=False,shuffleFlag=True)
# Shape of training set - 48999, 32, 32, 1
# Shape of test set - 5445, 32, 32, 1
############################################################

############################################################
# SECTION 2
# TRAIN YOUR MODEL. BUILD MODEL IN A SEPARATE FUNCTION FILE
############################################################
# buildModel.model_01(name='model_01',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=200,batchSize=128)
# buildModel.model_02(name='model_02',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=200,batchSize=128)
# buildModel.model_03(name='model_03',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=50,batchSize=128)
# buildModel.model_04(name='model_04',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=50,batchSize=128)
# buildModel.model_05(name='model_05',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=50,batchSize=128)
# buildModel.trainUsingVGG16(name='vgg16',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=20,batchSize=128)
############################################################


############################################################
# SECTION 3
# TRAIN SAVED MODEL WITH A DIFFERENT OPTIMIZATION PARAMTERS
############################################################
# optimizer = optimizers.SGD(learning_rate=0.001,momentum=0.99,nesterov=False)
# buildModel.trainIntermediateModel(modelFile='vgg16_20200107_intermediate_003_intermediate_020_intermediate_030_accuracy_trainAcc_99.77_testAcc_99.76',name='vgg16_20200107_intermediate_003_intermediate_020_intermediate_030',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,optimizer=optimizer,epochs=100,batchSize=128,bestModelMode='all')
############################################################


############################################################
# SECTION 4
# RUN THE MODEL AND ON TEST AND TRAIN DATASET AND GENERATE
# IMAGES FOR FALSE IDENTIFICATION
############################################################
trainImagesPath = '../dataset/incorrectClassifications/train'
testImagesPath = '../dataset/incorrectClassifications/test'

###### DNN model - model_01() in ../lib/buildModel.py
# modelPath = '../model/model_01_test_intermediate_086_intermediate_091_accuracy_trainAcc_99.42_testAcc_99.47.h5'
# print ('Classification using %s' %(modelPath))
# utils.falseClassificationImage(modelPath,trainImagesPath,xTrain,yTrain)
# utils.falseClassificationImage(modelPath,testImagesPath,xTest,yTest)

###### CNN model - model_02() in ../lib/buildModel.py
# modelPath = '../model/model_02_20200106_intermediate_025_intermediate_007_accuracy_trainAcc_99.84_testAcc_99.83.h5'
# print ('Classification using %s' %(modelPath))
# utils.falseClassificationImage(modelPath,trainImagesPath,xTrain,yTrain)
# utils.falseClassificationImage(modelPath,testImagesPath,xTest,yTest)

###### VGG16 model - trainUsingVGG16() in ../lib/buildModel.py
# modelPath = '../model/vgg16_20200107_intermediate_003_intermediate_020_intermediate_030_intermediate_099_accuracy_trainAcc_99.93_testAcc_99.93.h5'
# print ('Classification using %s' %(modelPath))
# utils.falseClassificationImage(modelPath,trainImagesPath,xTrain,yTrain)
# utils.falseClassificationImage(modelPath,testImagesPath,xTest,yTest)
############################################################

############################################################
# SECTION 5
# TEST THE ACCURACY FOR EACH OF THE FINAL SELECTED MODELS
############################################################
# utils.modelAccuracy('../model/model_01_test_intermediate_086_intermediate_091_accuracy_trainAcc_99.42_testAcc_99.47.h5',xTrain,yTrainInd,xTest,yTestInd)
# utils.modelAccuracy('../model/model_02_20200106_intermediate_025_intermediate_007_accuracy_trainAcc_99.84_testAcc_99.83.h5',xTrain,yTrainInd,xTest,yTestInd)
# utils.modelAccuracy('../model/vgg16_20200107_intermediate_003_intermediate_020_intermediate_030_intermediate_099_accuracy_trainAcc_99.93_testAcc_99.93.h5',xTrain,yTrainInd,xTest,yTestInd)
