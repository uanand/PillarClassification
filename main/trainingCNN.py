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
# LOAD THE LABELLED DATASET AND SPLIT INTO TRAINING AND TEST
############################################################
xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd = loadData.loadPillarData(fileName='../dataset/newLabelledDataset.dat',numClasses=2,row=32,col=32,rotFlag=True,flipFlag=True,RGB=True,shuffleFlag=True)
# xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd = loadData.loadPillarData(fileName='../dataset/labelledDataset.dat',numClasses=2,row=32,col=32,rotFlag=True,flipFlag=True,RGB=True,shuffleFlag=True)
print (xTrain.shape,yTrainInd.shape,xTest.shape,yTestInd.shape,numpy.sum(yTrainInd,axis=0),numpy.sum(yTestInd,axis=0))

# for i in range(10):
    # # plt.figure(), plt.imshow(xTrain[i,:,:,:]), plt.show()
    # plt.figure(), plt.imshow(xTrain[i,:,:,0]), plt.title(yTrain[i]), plt.show()
# Shape of training set - 48999, 32, 32, 1
# Shape of test set - 5445, 32, 32, 1
############################################################

############################################################
# TRAIN YOUR MODEL. BUILD MODEL IN A SEPARATE FUNCTION FILE
############################################################
# buildModel.model_01(name='model_01_test',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=200,batchSize=128)
# buildModel.model_02(name='model_02_20200106',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=200,batchSize=128)
# buildModel.model_03(name='model_03_20200106',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=50,batchSize=1000)
# buildModel.model_04(name='model_04_newData',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=50,batchSize=200)
# buildModel.model_05(name='model_05_newData',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=50,batchSize=200)
# buildModel.trainUsingVGG16(name='vgg16_20200107',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=20,batchSize=128)
############################################################


############################################################
# TRAIN SAVED MODEL WITH A DIFFERENT OPTIMIZATION PARAMTERS
############################################################
# optimizer = optimizers.Adadelta(learning_rate=0.001,rho=0.95,epsilon=1e-07)
# optimizer = optimizers.Adagrad(learning_rate=0.001,initial_accumulator_value=0.1,epsilon=1e-07)
# optimizer = optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07,amsgrad=False)
# optimizer = optimizers.Adamax(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07)
# optimizer = optimizers.Ftrl(learning_rate=0.001,learning_rate_power=-0.5,initial_accumulator_value=0.1,l1_regularization_strength=0.0,l2_regularization_strength=0.0)
# optimizer = optimizers.Nadam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07)
# optimizer = optimizers.RMSprop(learning_rate=0.001,rho=0.9,momentum=0.0,epsilon=1e-07,centered=False)
# optimizer = optimizers.SGD(learning_rate=0.001,momentum=0.99,nesterov=False)

# optimizer = optimizers.SGD(learning_rate=0.001,momentum=0.99,nesterov=False)
# buildModel.trainIntermediateModel(modelFile='vgg16_20200107_intermediate_003_intermediate_020_intermediate_030_accuracy_trainAcc_99.77_testAcc_99.76',name='vgg16_20200107_intermediate_003_intermediate_020_intermediate_030',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,optimizer=optimizer,epochs=100,batchSize=128,bestModelMode='all')
############################################################


############################################################
# RUN THE MODEL AND ON TEST AND TRAIN DATASET AND GENERATE
# IMAGES FOR FALSE IDENTIFICATION
############################################################
# trainImagesPath = '../dataset/incorrectClassifications/train'
# testImagesPath = '../dataset/incorrectClassifications/test'

# modelPath = '../model/model_01_test_intermediate_086_intermediate_091_accuracy_trainAcc_99.42_testAcc_99.47.h5'
# print ('Classification using %s' %(modelPath))
# utils.falseClassificationImage(modelPath,trainImagesPath,xTrain,yTrain)
# utils.falseClassificationImage(modelPath,testImagesPath,xTest,yTest)

# modelPath = '../model/model_02_20200106_intermediate_025_intermediate_007_accuracy_trainAcc_99.84_testAcc_99.83.h5'
# print ('Classification using %s' %(modelPath))
# utils.falseClassificationImage(modelPath,trainImagesPath,xTrain,yTrain)
# utils.falseClassificationImage(modelPath,testImagesPath,xTest,yTest)

# modelPath = '../model/vgg16_20200107_intermediate_003_intermediate_020_intermediate_030_intermediate_099_accuracy_trainAcc_99.93_testAcc_99.93.h5'
# print ('Classification using %s' %(modelPath))
# utils.falseClassificationImage(modelPath,trainImagesPath,xTrain,yTrain)
# utils.falseClassificationImage(modelPath,testImagesPath,xTest,yTest)
############################################################

############################################################
# TEST THE ACCURACY FOR EACH OF THE FINAL SELECTED MODELS
############################################################
# utils.modelAccuracy('../model/model_01_test_intermediate_086_intermediate_091_accuracy_trainAcc_99.42_testAcc_99.47.h5',xTrain,yTrainInd,xTest,yTestInd)
# utils.modelAccuracy('../model/model_02_20200106_intermediate_025_intermediate_007_accuracy_trainAcc_99.84_testAcc_99.83.h5',xTrain,yTrainInd,xTest,yTestInd)
# utils.modelAccuracy('../model/vgg16_20200107_intermediate_003_intermediate_020_intermediate_030_intermediate_099_accuracy_trainAcc_99.93_testAcc_99.93.h5',xTrain,yTrainInd,xTest,yTestInd)