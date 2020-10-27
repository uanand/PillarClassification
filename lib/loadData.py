import numpy
from sklearn.utils import shuffle

import transform

############################################################
# CONVERT Y TO AN INDICATOR MATRIX
############################################################
def y2indicator(y,numClasses):
    '''
    Convert classification labelled array y to indicator matrix.
    
    Input parameters:
    yTest : (1D array) Classification labels for N labelled images.
    numClasses : (int) Number of classes in the labelled dataset. For
        the nanopillar classification, it is 2.
    
    Returns:
    yInd : (2D array) Indicator matrix for the labelled dataset.
        Its shape is [N,numClasses].
    '''
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
    '''
    Load the training and test dataset into system memory and perform
    data augmentation based on user inputs.
    
    Input parameters:
    fileName : (str) Name with full path of the file which has labelled
        dataset. The first column corresponds to the label, and the
        other columns correspond to the intensity value of each pixel of
        64x64 images that are flattened usign numpy.flatten().
    numClasses : (int) Number of classes in the labelled dataset. For
        the nanopillar classification, it is 2. 
    row : (int) Desired image size for training neural network model.
    col : (int) Desired image size for training neural network model.
    rotFlag : (bool) Use rotation (90, 180, 270 deg) for data
        augmentation. Default is True.
    flipFlag : (bool) use flipping around y axis for data aurgmentation.
        Default is True.
    RGB: (bool) Number of channels in training and test images. If true,
        then each image will have 3 components. Turn on when using VGG16
        model. Default is False.
    shuffleFlag : (bool) Randomly shuffle the data before splitting into
    training and test datasets. Default is True.
    
    Returns:
    xTrain : (4D array) Normalized training dataset of shape
        [N,row,col,1]. Usually the intensity values of each image are
        normalized between 0 and 1.
    yTrain : (1D array) Classification label for each training image.
        The size of this array is N.
    yTrainInd : (2D array) Indicator matrix for the training dataset.
        Its shape is [N,numClasses].
    xTest : (4D array) Normalized training dataset of shape
        [N,row,col,1]. Usually the intensity values of each image are
        normalized between 0 and 1.
    yTest : (1D array) Classification label for each training image.
        The size of this array is N.
    yTestInd : (2D array) Indicator matrix for the training dataset.
        Its shape is [N,numClasses].
    '''
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
