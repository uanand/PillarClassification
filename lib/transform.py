import numpy
import cv2
from skimage.transform import rotate

import imageProcess

############################################################
# ROTATE THE TRAINING DATASET TO GENERATE MORE IMAGES FOR TRAINING
############################################################
def rotateDataset(x,y,angle):
    [rowDataset,colDataset] = x.shape
    row,col = int(numpy.sqrt(colDataset)),int(numpy.sqrt(colDataset))
    xRot = x.copy(); xRot[:] = 0;
    yRot = y.copy()
    gImgRot = numpy.zeros([row,col],dtype='uint8')
    for i in range(rowDataset):
        gImg = numpy.reshape(x[i,:],(row,col))
        gImgRot = numpy.round((rotate(gImg,angle)*255)).astype('uint8')
        xRot[i,:] = gImgRot.flatten()
    return xRot,yRot
############################################################

############################################################
# FLIP THE DATASET AROUND THE HORIZINTAL AND VERTICAL AXES
############################################################
def flipDataset(x,y):
    [rowDataset,colDataset] = x.shape
    row,col = int(numpy.sqrt(colDataset)),int(numpy.sqrt(colDataset))
    xFlipHor,yFlipHor = x.copy(),y.copy(); xFlipHor[:] = 0;
    xFlipVer,yFlipVer = x.copy(),y.copy(); xFlipVer[:] = 0;
    for i in range(rowDataset):
        gImg = numpy.reshape(x[i,:],(row,col))
        gImgFlipHor = numpy.flip(gImg,axis=1)
        gImgFlipVer = numpy.flip(gImg,axis=0)
        xFlipHor[i,:] = gImgFlipHor.flatten()
        xFlipVer[i,:] = gImgFlipVer.flatten()
    xFlip = numpy.row_stack((xFlipHor,xFlipVer))
    yFlip = numpy.concatenate((yFlipHor,yFlipVer))
    return xFlip,yFlip
############################################################

############################################################
# RESIZE THE TRAINING DATASET
############################################################
def resizeDataset(x,size):
    [rowDataset,colDataset] = x.shape
    row,col = int(numpy.sqrt(colDataset)),int(numpy.sqrt(colDataset))
    xResize = numpy.zeros([rowDataset,size[0]*size[1]],dtype='uint8')
    for i in range(rowDataset):
        gImg = numpy.reshape(x[i,:],(row,col))
        gImgResize = cv2.resize(gImg,size,interpolation=cv2.INTER_LINEAR)
        xResize[i,:] = gImgResize.flatten()
    return xResize
############################################################

############################################################
# CONVERT THE DATASET TO RGB FORMAT
############################################################
def convetToRGB(xTrain,xTest):
    xTrainRGB = numpy.zeros([xTrain.shape[0],xTrain.shape[1],xTrain.shape[2],3])
    xTestRGB = numpy.zeros([xTest.shape[0],xTest.shape[1],xTest.shape[2],3])
    for i in range(xTrain.shape[0]):
        img = xTrain[i,:,:,0]
        xTrainRGB[i,:,:,0] = img
        xTrainRGB[i,:,:,1] = img
        xTrainRGB[i,:,:,2] = img
    for i in range(xTest.shape[0]):
        img = xTest[i,:,:,0]
        xTestRGB[i,:,:,0] = img
        xTestRGB[i,:,:,1] = img
        xTestRGB[i,:,:,2] = img
    return xTrainRGB,xTestRGB
############################################################

############################################################
# RENORMALIZE DATASET TO DIFFERENT MODEL FORMATS
############################################################
def renormalizeDataset(xTrain,xTest,VGG):
    if (VGG==True):
        try:
            [NTrain,rowTrain,colTrain,channelTrain] = xTrain.shape
            [NTest,rowTest,colTest,channelTest] = xTest.shape
            RGBFlag = True
        except:
            [NTrain,rowTrain,colTrain,channelTrain] = xTrain.shape
            [NTest,rowTest,colTest,channelTest] = xTest.shape
            RGBFlag = False
            
        xTrain *= 255; xTest *= 255
        for i in range(NTrain):
            if (RGBFlag==True):
                rMean = numpy.mean(xTrain[i,:,:,0])
                gMean = numpy.mean(xTrain[i,:,:,1])
                bMean = numpy.mean(xTrain[i,:,:,2])
                xTrain[i,:,:,0] -= rMean
                xTrain[i,:,:,1] -= gMean
                xTrain[i,:,:,2] -= bMean
            elif (RGBFlag==False):
                mean = numpy,mean(xTrain[i,:,:])
                xTrain[i,:,:] -= mean
        for i in range(NTest):
            if (RGBFlag==True):
                rMean = numpy.mean(xTest[i,:,:,0])
                gMean = numpy.mean(xTest[i,:,:,1])
                bMean = numpy.mean(xTest[i,:,:,2])
                xTest[i,:,:,0] -= rMean
                xTest[i,:,:,1] -= gMean
                xTest[i,:,:,2] -= bMean
            elif (RGBFlag==False):
                mean = numpy,mean(xTest[i,:,:])
                xTest[i,:,:] -= mean
    return xTrain,xTest
############################################################

############################################################
# NORMALIZE EACH IMAGE BETWEEN 0-255
############################################################
def normalizeDataset(x):
    try:
        [rowDataset,colDataset] = x.shape
        for i in range(rowDataset):
            x[i,:] = imageProcess.normalize(x[i,:])
    except:
        [N,row,col,channel] = x.shape
        for i in range(N):
            x[i,:,:,:] = imageProcess.normalize(x[i,:,:,:])
    return x
############################################################
