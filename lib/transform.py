import numpy
import cv2
from skimage.transform import rotate
import matplotlib.pyplot as plt

#######################################################################
# ROTATE THE TRAINING DATASET TO GENERATE MORE IMAGES FOR TRAINING
#######################################################################
def rotateDataset(x,y,angle):
    [rowDataset,colDataset] = x.shape
    row,col = int(numpy.sqrt(colDataset)),int(numpy.sqrt(colDataset))
    xRot = x.copy(); xRot[:] = 0;
    yRot = y.copy()
    for i in range(rowDataset):
        gImg = numpy.reshape(x[i,:],(row,col))
        gImgRot = rotate(gImg,angle)
        xRot[i,:] = gImgRot.flatten()
    return xRot,yRot
#######################################################################


#######################################################################
# FLIP THE DATASET AROUND THE HORIZINTAL AND VERTICAL AXES
#######################################################################
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
#######################################################################


#######################################################################
# RESIZE THE TRAINING DATASET
#######################################################################
def resizeDataset(x,size):
    [rowDataset,colDataset] = x.shape
    row,col = int(numpy.sqrt(colDataset)),int(numpy.sqrt(colDataset))
    xResize = numpy.zeros([rowDataset,size[0]*size[1]],dtype='uint8')
    for i in range(rowDataset):
        gImg = numpy.reshape(x[i,:],(row,col))
        # gImgResize = cv2.resize(gImg,size,interpolation=cv2.INTER_AREA)
        gImgResize = cv2.resize(gImg,size,interpolation=cv2.INTER_LINEAR)
        xResize[i,:] = gImgResize.flatten()
    return xResize
#######################################################################


#######################################################################
# CONVERT THE DATASET TO RGB FORMAT
#######################################################################
def convetToRGB(xTrain,xTest):
    xTrainRGB = numpy.zeros([xTrain.shape[0],xTrain.shape[1],xTrain.shape[2],3],dtype='uint8')
    xTestRGB = numpy.zeros([xTest.shape[0],xTest.shape[1],xTest.shape[2],3],dtype='uint8')
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
#######################################################################
