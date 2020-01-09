import numpy
import hyperspy.api as hs
from scipy import ndimage

############################################################
# NORMALIZE AN 8 BIT GRAYSCALE IMAGE
############################################################
def normalize(gImg, min=0, max=255):
    if (gImg.max() > gImg.min()):
        gImg = 1.0*(max-min)*(gImg - gImg.min())/(gImg.max() - gImg.min())
        gImg=gImg+min
    elif (gImg.max() > 0):
        gImg[:] = max
    gImg=gImg.astype('uint8')
    return gImg
############################################################

############################################################
# READ GATAN DM3 OR DM4 FILES
############################################################
def readDM4(fileName):
    f = hs.load(fileName);
    gImg = f.data
    lowLimit = f.original_metadata['DocumentObjectList']['TagGroup0']['ImageDisplayInfo']['LowLimit']
    highLimit = f.original_metadata['DocumentObjectList']['TagGroup0']['ImageDisplayInfo']['HighLimit']
    gImg[gImg<lowLimit] = lowLimit
    gImg[gImg>highLimit] = highLimit
    gImg = normalize(gImg)
    return gImg
############################################################

############################################################
# REMOVE BINARY WHITE PARTICLES TOUCHING THE BOUNDARY
############################################################
def removeBoundaryParticles(bImg):
    [row,col] = bImg.shape
    labelImg,numLabel = ndimage.label(bImg)
    boundaryLabels = numpy.unique(numpy.concatenate((numpy.unique(labelImg[:,0]),numpy.unique(labelImg[:,-1]),numpy.unique(labelImg[0,:]),numpy.unique(labelImg[-1,:]))))
    boundaryLabels = boundaryLabels[boundaryLabels>0]
    for label in boundaryLabels:
        labelImg[labelImg==label] = 0
    return labelImg.astype('bool')
############################################################
