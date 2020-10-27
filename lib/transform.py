import numpy
import cv2
from skimage.transform import rotate

import imageProcess

############################################################
# ROTATE THE TRAINING DATASET TO GENERATE MORE IMAGES FOR TRAINING
############################################################
def rotateDataset(x,y,angle):
    '''
    Rotate the dataset by a certain angle for data augmentation. 
    
    Input parameters:
    x : (2D array) With N rows and 1024 or 4096 columns. Each row
        corresponds to a flattened image.
    y : (1D array) Classification tags for N labelled images.
    angle : (double) Rotation angle in degrees in counter-clockwise
        direction.
    
    Returns:
    xRot : (2D array) Rotated and flattened images with N rows and 1024
        or 4096 columns. Each row corresponds to a flattened image.
    yRot : (1D array) Classification tags for N rotated images. Same as
        y.
    '''
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
    '''
    Flip the dataset around the vertical central axis. 
    
    Input parameters:
    x : (2D array) With N rows and 1024 or 4096 columns. Each row
        corresponds to a flattened image.
    y : (1D array) Classification tags for N labelled images.
    
    Returns:
    xFlip : (2D array) Flipped and flattened images with N rows and 1024
        or 4096 columns. Each row corresponds to a flattened image.
    yRot : (1D array) Classification tags for N flipped images. Same as
        y.
    '''
    [rowDataset,colDataset] = x.shape
    row,col = int(numpy.sqrt(colDataset)),int(numpy.sqrt(colDataset))
    xFlipHor,yFlipHor = x.copy(),y.copy(); xFlipHor[:] = 0;
    for i in range(rowDataset):
        gImg = numpy.reshape(x[i,:],(row,col))
        gImgFlipHor = numpy.flip(gImg,axis=1)
        xFlipHor[i,:] = gImgFlipHor.flatten()
    xFlip = xFlipHor.copy()
    yFlip = yFlipHor.copy()
    return xFlip,yFlip
############################################################

############################################################
# RESIZE THE TRAINING DATASET
############################################################
def resizeDataset(x,size):
    '''
    Resize the dataset from AxA pixels to BxB pixels. 
    
    Input parameters:
    x : (2D array) With N rows and 1024 or 4096 columns. Each row
        corresponds to a flattened image.
    size : (tuple/list) Final desired size of each labelled image. Here,
        the labelled dataset was 64x64 pixels, and we resized it to
        32x32 pixels. size = [32,32], or size = (32,32).
        
    Returns:
    xResize : (2D array) Resized and flattened images with N rows. Each
        row corresponds to a flattened image.
    '''
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
    '''
    Convert the training and test datasets from single channel grayscale
    to 3 channel RGB.
    
    Input parameters:
    xTrain : (4D array) Training dataset of shape [N,row,col,1]. Here N
        is the number of images in the training dataset. 
    xTest : (4D array) Test dataset of shape [N,row,col,1]. Here N
        is the number of images in the test dataset.
    
    Returns:
    xTrain : (4D array) Training dataset of shape [N,row,col,3],
        corresponding to R, G, and B channels.
    xTest : (4D array) Test dataset of shape [N,row,col,3],
        corresponding to R, G, and B channels.
    '''
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
    '''
    Renormalize the training and test dataset as was done in the
    original article. DOI - https://arxiv.org/abs/1409.1556v6
    
    Input parameters:
    xTrain : (4D array) Training dataset of shape [N,row,col,1(3)]. Here
        N is the number of images in the training dataset. 
    xTest : (4D array) Test dataset of shape [N,row,col,1(3)]. Here N
        is the number of images in the test dataset.
    VGG : (bool) Flag to perform this renormalization.
    
    Returns:
    xTrain : (4D array) Renormalized training dataset of shape
        [N,row,col,3].
    xTest : (4D array) Renormalized test dataset of shape [N,row,col,3].
    '''
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
    '''
    Stretch the contrast of each labelled dataset between 0 to 255.
    
    Input parameters:
    x : (2D array) With N rows and 1024 or 4096 columns. Each row
        corresponds to a flattened image.
        
    Returns:
    x : (2D array) With N rows and 1024 or 4096 columns. Each row
        corresponds to a flattened image. The intensity values of each
        image are stretched between 0 to 255.
    '''
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
