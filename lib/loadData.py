import numpy
import transform
from sklearn.utils import shuffle

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
def loadPillarData(fileName,numClasses):
    labelledDataset = numpy.loadtxt(fileName,skiprows=1)
    [numLabelledDataset,temp] = labelledDataset.shape
    
    xOriginal,yOriginal = labelledDataset[:,1:],labelledDataset[:,0]
    xRot090,yRot090 = transform.rotateDataset(xOriginal,yOriginal,90)
    xRot180,yRot180 = transform.rotateDataset(xOriginal,yOriginal,180)
    xRot270,yRot270 = transform.rotateDataset(xOriginal,yOriginal,270)
    xFlipHorizontal,yFlipHorizontal = transform.flipDataset()
    xFlipVertical,yFlipVertical = transform.flipDataset()
    
    
    
    x_train_original,y_train_original = labelledDataset[:4000,1:],labelledDataset[:4000,0]
    x_test_original,y_test_original = labelledDataset[4000:,1:],labelledDataset[4000:,0]
    x_train_rot90,y_train_rot90 = transform.rotateDataset(x_train_original,y_train_original,90)
    x_test_rot90,y_test_rot90 = transform.rotateDataset(x_test_original,y_test_original,90)
    x_train_rot180,y_train_rot180 = transform.rotateDataset(x_train_original,y_train_original,180)
    x_test_rot180,y_test_rot180 = transform.rotateDataset(x_test_original,y_test_original,180)
    x_train_rot270,y_train_rot270 = transform.rotateDataset(x_train_original,y_train_original,270)
    x_test_rot270,y_test_rot270 = transform.rotateDataset(x_test_original,y_test_original,270)
    
    x_train = numpy.row_stack((x_train_original,x_train_rot90,x_train_rot180,x_train_rot270))
    y_train = numpy.concatenate((y_train_original,y_train_rot90,y_train_rot180,y_train_rot270))
    x_test = numpy.row_stack((x_test_original,x_test_rot90,x_test_rot180,x_test_rot270))
    y_test = numpy.concatenate((y_test_original,y_test_rot90,y_test_rot180,y_test_rot270))
    
    x_train = transform.resizeDataset(x_train,(32,32))
    x_test = transform.resizeDataset(x_test,(32,32))
    x_train = numpy.reshape(x_train,(x_train.shape[0],32,32,1))
    x_test = numpy.reshape(x_test,(x_test.shape[0],32,32,1))
    x_train,x_test = x_train.astype('float32'),x_test.astype('float32')
    x_train/=255; x_test/=255
    y_train,y_test = y_train.astype('uint8'),y_test.astype('uint8')
    
    y_train_ind = y2indicator(y_train,numClasses)
    y_test_ind = y2indicator(y_test,numClasses)
    
    x_train,y_train,y_train_ind = shuffle(x_train,y_train,y_train_ind)
    x_test,y_test,y_test_ind = shuffle(x_test,y_test,y_test_ind)
    return x_train,y_train,y_train_ind,x_test,y_test,y_test_ind
############################################################


# numpy.random.seed(0)
# numpy.random.shuffle(labelledDataset)
# [numLabelledDataset,temp] = labelledDataset.shape

# x_train_original,y_train_original = labelledDataset[:4000,1:],labelledDataset[:4000,0]
# x_test_original,y_test_original = labelledDataset[4000:,1:],labelledDataset[4000:,0]
# x_train_rot90,y_train_rot90 = transform.rotateDataset(x_train_original,y_train_original,90)
# x_test_rot90,y_test_rot90 = transform.rotateDataset(x_test_original,y_test_original,90)
# x_train_rot180,y_train_rot180 = transform.rotateDataset(x_train_original,y_train_original,180)
# x_test_rot180,y_test_rot180 = transform.rotateDataset(x_test_original,y_test_original,180)
# x_train_rot270,y_train_rot270 = transform.rotateDataset(x_train_original,y_train_original,270)
# x_test_rot270,y_test_rot270 = transform.rotateDataset(x_test_original,y_test_original,270)

# x_train = numpy.row_stack((x_train_original,x_train_rot90,x_train_rot180,x_train_rot270))
# y_train = numpy.concatenate((y_train_original,y_train_rot90,y_train_rot180,y_train_rot270))
# x_test = numpy.row_stack((x_test_original,x_test_rot90,x_test_rot180,x_test_rot270))
# y_test = numpy.concatenate((y_test_original,y_test_rot90,y_test_rot180,y_test_rot270))

# x_train = transform.resizeDataset(x_train,(32,32))
# x_test = transform.resizeDataset(x_test,(32,32))
# row,col = 32,32

# x_train = numpy.reshape(x_train,(x_train.shape[0],row,col,1))
# x_test = numpy.reshape(x_test,(x_test.shape[0],row,col,1))
# x_train,x_test = x_train.astype('float32'),x_test.astype('float32')
# x_train/=255; x_test/=255

# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

# num_classes = y_test.shape[1]
# num_pixels = x_train.shape[1] * x_train.shape[2]
# num_trainData,num_testData = x_train.shape[0],x_test.shape[0]
# ############################################################
