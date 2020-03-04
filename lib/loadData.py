import numpy
from sklearn.utils import shuffle

import transform

############################################################
# CONVERT Y TO AN INDICATOR MATRIX
############################################################
def y2indicator(y,numClasses):
    N = y.size
    yInd = numpy.zeros([N,numClasses],dtype='float32')
    for i in range(N):
        yInd[i,y[i]] = 1
    return yInd
############################################################

############################################################
# LOAD THE LABELLED PILLAR DATASET AND SPLIT INTO TRAINING AND TEST
############################################################
def loadPillarData(fileName,numClasses,row=32,col=32,rotFlag=True,flipFlag=True,RGB=False,shuffleFlag=True):
    labelledDataset = numpy.loadtxt(fileName,skiprows=1,dtype='uint8')
    [numLabelledDataset,temp] = labelledDataset.shape
    
    xOriginal,yOriginal = labelledDataset[:,1:],labelledDataset[:,0]
    if (rotFlag==True):
        xRot090,yRot090 = transform.rotateDataset(xOriginal,yOriginal,90)
        xRot180,yRot180 = transform.rotateDataset(xOriginal,yOriginal,180)
        xRot270,yRot270 = transform.rotateDataset(xOriginal,yOriginal,270)
        xOriginal = numpy.row_stack((xOriginal,xRot090,xRot180,xRot270))
        yOriginal = numpy.concatenate((yOriginal,yRot090,yRot180,yRot270))
    if (flipFlag==True):
        xFlip,yFlip = transform.flipDataset(xOriginal,yOriginal)
        xOriginal = numpy.row_stack((xOriginal,xFlip))
        yOriginal = numpy.concatenate((yOriginal,yFlip))
        
    [rowDataset,colDataset] = xOriginal.shape
    if (shuffleFlag==True):
        xOriginal,yOriginal = shuffle(xOriginal,yOriginal,random_state=1)
    xTrain,yTrain = xOriginal[:int(0.9*rowDataset),:],yOriginal[:int(0.9*rowDataset)]
    xTest,yTest = xOriginal[int(0.9*rowDataset):,:],yOriginal[int(0.9*rowDataset):]
    
    xTrain = transform.resizeDataset(xTrain,(row,col))
    xTest = transform.resizeDataset(xTest,(row,col))
    xTrain = numpy.reshape(xTrain,(xTrain.shape[0],row,col,1))
    xTest = numpy.reshape(xTest,(xTest.shape[0],row,col,1))
    xTrain = transform.normalizeDataset(xTrain)
    xTest = transform.normalizeDataset(xTest)
    
    if (RGB==True):
        xTrain,xTest = transform.convetToRGB(xTrain,xTest)
    xTrain,xTest = xTrain.astype('float32'),xTest.astype('float32')
    xTrain/=255.0; xTest/=255.0
    yTrain,yTest = yTrain.astype('uint8'),yTest.astype('uint8')
    
    yTrainInd = y2indicator(yTrain,numClasses)
    yTestInd = y2indicator(yTest,numClasses)
    
    if (shuffleFlag==True):
        xTrain,yTrain,yTrainInd = shuffle(xTrain,yTrain,yTrainInd,random_state=1)
        xTest,yTest,yTestInd = shuffle(xTest,yTest,yTestInd,random_state=1)
        
    return xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd
############################################################
