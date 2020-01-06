import os
import sys
import numpy
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras import optimizers

sys.path.append(os.path.abspath('../lib'))
import loadData
import buildModel
import utils

############################################################
# LOAD THE LABELLED DATASET AND SPLIT INTO TRAINING AND TEST
############################################################
# xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd = loadData.loadPillarData(fileName='../dataset/newLabelledDataset.dat',numClasses=2,row=32,col=32,rotFlag=True,flipFlag=True,RGB=False)
xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd = loadData.loadPillarData(fileName='../dataset/labelledDataset.dat',numClasses=2,row=32,col=32,rotFlag=True,flipFlag=True,RGB=False)
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
buildModel.model_02(name='model_02_test',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=200,batchSize=128)
# buildModel.model_03(name='model_03_fullData',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=2,batchSize=32)
# buildModel.model_04(name='model_04_newData',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=50,batchSize=200)
# buildModel.model_05(name='model_05_newData',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=50,batchSize=200)
# buildModel.trainUsingVGG16(name='vgg16_oldData',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=20,batchSize=32)
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

# optimizer = optimizers.SGD(learning_rate=0.01,momentum=0.99,nesterov=False)
# buildModel.trainIntermediateModel(modelFile='model_02_partialData_epochs_20_batchsize_32_trainAcc_98.41_testAcc_98.02',name='model_02_fullData',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,optimizer=optimizer,epochs=50,batchSize=32,bestModelMode='all')
############################################################


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
