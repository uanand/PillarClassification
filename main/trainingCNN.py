import os
import sys
from tqdm import tqdm
import numpy
import cv2
from tensorflow import keras

sys.path.append(os.path.abspath('../lib'))
import loadData
import buildModel

############################################################
# LOAD THE LABELLED DATASET AND SPLIT INTO TRAINING AND TEST
############################################################
xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd = loadData.loadPillarData(fileName='../dataset/labelledDataset.dat',numClasses=2,rotFlag=True,flipFlag=True)
# Shape of training set - 48999, 32, 32, 1
# Shape of test set - 5445, 32, 32, 1
############################################################


############################################################
# TRAIN YOUR MODEL. BUILD MODEL IN A SEPARATE FUNCTION FILE
############################################################
buildModel.model_01(name='model_01',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=500,batchSize=1000)
buildModel.model_02(name='model_02',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=200,batchSize=1000)
buildModel.model_03(name='model_03',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=200,batchSize=1000)
buildModel.model_04(name='model_04',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=200,batchSize=1000)
# buildModel.model_05(name='model_05',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=200,batchSize=1000)
# buildModel.model_06(name='model_06',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=200,batchSize=1000)
############################################################


############################################################
# RUN THE MODEL AND ON TEST AND TRAIN DATASET AND GENERATE
# CORRESPONDING IMAGES
############################################################

# model = keras.models.load_model('/home/utkarsh/Projects/PillarClassification/model/model_01_intermediate_476_accuracy_trainAcc_99.78_testAcc_99.63.h5')
# counter = 1
# for i,(x,y) in enumerate(zip(xTrain,yTrain)):
    # gImg = numpy.reshape(x,(1,32,32,1))
    # gImgRaw = (numpy.reshape(gImg,(32,32))*255).astype('uint8')
    # if (y==0):
        # assignedLabel = 'Collapse'
    # else:
        # assignedLabel = 'Not collapse'
    # res = model.predict_classes(gImg,batch_size=1)[0]
    # keras.backend.clear_session()
    # if (res==0):
        # predictLabel = 'Collapse'
    # elif (res==1):
        # predictLabel = 'Not collapse'
    # if (assignedLabel!=predictLabel):
        # print('trainData',i+1,assignedLabel,predictLabel)
        # cv2.imwrite('/home/utkarsh/Projects/PillarClassification/dataset/incorrectClassifications/train/'+str(i+1).zfill(6)+'.png',gImgRaw)
        # counter+=1
# print (xTrain.shape)
        
# counter = 1
# for x,y in zip(xTest,yTest):
    # gImg = numpy.reshape(x,(1,32,32,1))
    # gImgRaw = (numpy.reshape(gImg,(32,32))*255).astype('uint8')
    # if (y==0):
        # assignedLabel = 'Collapse'
    # else:
        # assignedLabel = 'Not collapse'
    # res = model.predict_classes(gImg,batch_size=1)[0]
    # keras.backend.clear_session()
    # if (res==0):
        # predictLabel = 'Collapse'
    # elif (res==1):
        # predictLabel = 'Not collapse'
    # if (assignedLabel!=predictLabel):
        # print('testData',counter,assignedLabel,predictLabel)
        # cv2.imwrite('/home/utkarsh/Projects/PillarClassification/dataset/incorrectClassifications/test/'+str(counter).zfill(6)+'.png',gImgRaw)
        # counter+=1
############################################################
