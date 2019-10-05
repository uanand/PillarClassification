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
# RESIZE THE TRAINING DATASET
#######################################################################
def resizeDataset(x,size):
    [rowDataset,colDataset] = x.shape
    row,col = int(numpy.sqrt(colDataset)),int(numpy.sqrt(colDataset))
    # rowNew,colNew = size(0),size[1]
    xResize = numpy.zeros([rowDataset,size[0]*size[1]],dtype='uint8')
    for i in range(rowDataset):
        gImg = numpy.reshape(x[i,:],(row,col))
        gImgResize = cv2.resize(gImg,size,interpolation=cv2.INTER_AREA)
        xResize[i,:] = gImgResize.flatten()
    return xResize
#######################################################################
