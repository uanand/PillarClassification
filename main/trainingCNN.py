import os
import sys
# import numpy
# import cv2
# from tensorflow import keras
# from tensorflow.keras import layers,optimizers
# import keras
# import matplotlib.pyplot as plt
# from keras.utils import np_utils
# from keras.models import Sequential,load_model
# from keras.layers import Dense,Dropout,Flatten,Activation
# from keras.layers import Conv2D,MaxPooling2D
# from keras import backend as K
# from keras.optimizers import SGD

sys.path.append(os.path.abspath('../lib'))
import loadData
import buildModel

############################################################
# LOAD THE LABELLED DATASET AND SPLIT INTO TRAINING AND TEST
############################################################
xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd = loadData.loadPillarData(fileName='../dataset/labelledDataset.dat',numClasses=2)
# Shape of training set - 48999, 32, 32, 1
# Shape of test set - 5445, 32, 32, 1
############################################################

buildModel.model_01(name='model_01',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=500,batchSize=1000)
# buildModel.model_02(name='model_02',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=200,batchSize=1000)
# buildModel.model_03(name='model_03',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=200,batchSize=1000)
# buildModel.model_04(name='model_04',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=200,batchSize=1000)
# buildModel.model_05(name='model_05',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=200,batchSize=1000)
# buildModel.model_06(name='model_06',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=200,batchSize=1000)


############################################################
# RUN THE MODEL ON TRAIN AND TEST DATASETS
############################################################
# model = load_model('../model/model2_batchsize_1000_epochs_200.h5')
# counter = 1
# for i in range(num_trainData/4):
    # gImg = x_train[i,:,:,:]
    # gImg = numpy.reshape(gImg,(1,32,32,1))
    # gImgRaw = (numpy.reshape(gImg,(32,32))*255).astype('uint8')
    
    # if (y_train[i,0]==1):
        # assignedLabel = 'Collapse'
    # else:
        # assignedLabel = 'Not collapse'
    # res = model.predict_classes(gImg,batch_size=1)[0]
    # if (res==0):
        # predictLabel = 'Collapse'
    # elif (res==1):
        # predictLabel = 'Not collapse'
    # if (assignedLabel!=predictLabel):
        # print('trainData',counter,i,assignedLabel,predictLabel)
        # counter+=1
        # cv2.imwrite('/home/utkarsh/Projects/PillarClassification/dataset/incorrectClassifications/train/'+str(i).zfill(6)+'.png',gImgRaw)
    
# counter = 1
# for i in range(num_testData/4):
    # gImg = x_test[i,:,:,:]
    # gImg = numpy.reshape(gImg,(1,32,32,1))
    # gImgRaw = (numpy.reshape(gImg,(32,32))*255).astype('uint8')
    # if (y_test[i,0]==1):
        # assignedLabel = 'Collapse'
    # else:
        # assignedLabel = 'Not collapse'
    # res = model.predict_classes(gImg,batch_size=1)[0]
    # if (res==0):
        # predictLabel = 'Collapse'
    # elif (res==1):
        # predictLabel = 'Not collapse'
    # if (assignedLabel!=predictLabel):
        # print('testData',counter,i,assignedLabel,predictLabel)
        # counter+=1
        # cv2.imwrite('/home/utkarsh/Projects/PillarClassification/dataset/incorrectClassifications/test/'+str(i).zfill(6)+'.png',gImgRaw)
############################################################
