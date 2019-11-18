import numpy
import cv2
import hyperspy.api as hs
from scipy import ndimage
from skimage.morphology import disk, white_tophat
from skimage import measure

import imageProcess

#######################################################################
# NORMALIZE AN 8 BIT GRAYSCALE IMAGE
#######################################################################
def normalize(gImg, min=0, max=255):
    if (gImg.max() > gImg.min()):
        gImg = 1.0*(max-min)*(gImg - gImg.min())/(gImg.max() - gImg.min())
        gImg=gImg+min
    elif (gImg.max() > 0):
        gImg[:] = max
    gImg=gImg.astype('uint8')
    return gImg
#######################################################################

#######################################################################
# READ GATAN DM3 OR DM4 FILES
#######################################################################
def readDM4(fileName):
    f = hs.load(fileName);
    gImg = f.data
    lowLimit = f.original_metadata['DocumentObjectList']['TagGroup0']['ImageDisplayInfo']['LowLimit']
    highLimit = f.original_metadata['DocumentObjectList']['TagGroup0']['ImageDisplayInfo']['HighLimit']
    gImg[gImg<lowLimit] = lowLimit
    gImg[gImg>highLimit] = highLimit
    gImg = imageProcess.normalize(gImg)
    return gImg
#######################################################################
