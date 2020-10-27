import numpy
import hyperspy.api as hs
from scipy import ndimage
from skimage import measure

############################################################
# NORMALIZE AN 8 BIT GRAYSCALE IMAGE
############################################################
def normalize(gImg, min=0, max=255):
    '''
    Linear contrast stretching.
    
    Input parameters:
    gImg : (numpy array) Image numpy array.
    min : (uint8) Minimum intensity value required for contrast adjusted
        image. Default is 0.
    min : (uint8) Maximum intensity value required for contrast adjusted
        image. Default is 255.
        
    Returns:
    gImg : (uint8 array) Contrast adjusted image. The minimum and
        maximum intensity values in this image are min and max
        respectively. All the intermediate intensity values are linearly
        adjusted.
    '''
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
    '''
    Load DM3/DM4 files in system memory as a uint8 2D image.
    
    Input parameters:
    fileName : (str) Complete file name with path of the DM3/DM4 file.
    
    Returns:
    gImg : (uint8 array) Contrast adjusted image. The minimum and
        maximum intensity values in this image are 0 and 255
        respectively. All the intermediate intensity values are linearly
        adjusted.
    '''
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
    '''
    Lable connected regions in a binary image and remove the labels at
    the boundary.
    
    Input parameters:
    bImg : (bool array) Boolean 2D array of binary image.
    
    Returns:
    bImg : (bool array) A new binary image where all the connected
        components touching the boundary have been removed. 
    '''
    [row,col] = bImg.shape
    labelImg,numLabel = ndimage.label(bImg)
    boundaryLabels = numpy.unique(numpy.concatenate((numpy.unique(labelImg[:,0]),numpy.unique(labelImg[:,-1]),numpy.unique(labelImg[0,:]),numpy.unique(labelImg[-1,:]))))
    boundaryLabels = boundaryLabels[boundaryLabels>0]
    for label in boundaryLabels:
        labelImg[labelImg==label] = 0
    return labelImg.astype('bool')
############################################################

############################################################
# CONVERT THE IMAGE TO RGB FORMAT
############################################################
def convetToRGB(gImg):
    '''
    Convert a grayscale 2D image to 3 channel RGB image.
    
    Input parameters:
    gImg : (2D array) Grayscale image.
    
    Returns:
    RGBImage : (3D array) RGB image with same values for the red, green,
        and blue channels. The shape of the array is [row, columns, 3]
        where 3 corresponds to RGB channels. First channel is red,
        second blue, and third green.  
    '''
    [row,col] = numpy.shape(gImg)
    gImgRGB = numpy.zeros([row,col,3],dtype=gImg.dtype)
    gImgRGB[:,:,0] = gImg
    gImgRGB[:,:,1] = gImg
    gImgRGB[:,:,2] = gImg
    return gImgRGB
############################################################

############################################################
# INVERT THE IMAGE
############################################################
def invert(gImg):
    '''
    Inverse transformation of a uint8 grayscale/RGB image.
    
    Input parameters:
    gImg : (uint8 array) Grayscale or RGB input image.
    
    Returns:
    gImgIng : (uint8 array) Inverted image where the intensity of all
        the pixels have been inverted. 
    '''
    gImgInv = 255-gImg
    return gImgInv
############################################################

############################################################
# OTSU THRESHOLD FOR GRAYSCALE IMAGES
############################################################
def otsuThreshold(img):
    '''
    Compute the Otsu's threshold for grayscale image. The algorithm is
    described here : DOI - 10.1109/TSMC.1979.4310076
    
    Input parameters:
    gImg : (uint8 array) Grayscale input image.
    
    Returns:
    threshold : (int) Intensity value at which the inter-class variance
        is minimized. 
    '''
    hist,bins = numpy.histogram(img.flatten(),bins=256,range=(0,255))
    totalPixels = hist.sum()
    
    currentMax = 0
    threshold = 0
    sumTotal, sumForeground, sumBackground = 0., 0., 0.
    weightBackground, weightForeground = 0., 0.
    
    for i,t in enumerate(hist):
        sumTotal += i * hist[i]
    for i,t in enumerate(hist):
        weightBackground += hist[i]
        if( weightBackground == 0 ):
            continue
        weightForeground = totalPixels - weightBackground
        if ( weightForeground == 0 ):
            break
            
        sumBackground += i*hist[i]
        sumForeground = sumTotal - sumBackground
        meanB = sumBackground / weightBackground
        meanF = sumForeground / weightForeground
        varBetween = weightBackground*weightForeground
        varBetween *= (meanB-meanF)*(meanB-meanF)
        if(varBetween > currentMax):
            currentMax = varBetween
            threshold = i
    return threshold
############################################################

############################################################
# KAPUR THRESHOLD FOR GRAYSCALE IMAGES
############################################################
def threshold_kapur(gImg):
    '''
    Compute the Kapur's threshold (also call entropy-maximizing
    threshold) for grayscale image. The algorithm is described here :
    DOI - 10.1016/0734-189X(85)90125-2
    
    Input parameters:
    gImg : (uint8 array) Grayscale input image.
    
    Returns:
    threshold : (int) Intensity value at which the inter-class entropy
        is maximized. 
    '''
    gImg = gImg.flatten()
    freq, bin_edges = numpy.histogram(intensities,bins=range(0,256),normed=False)
    total_freq = numpy.sum(freq)
    for s in range(256):
        entropyLeft=0
        entropyRight=0
        leftFreq = 0
        for i in range(s+1):
            leftFreq += freq[i]
        rightFreq = totalFreq-leftFreq
        
        if (leftFreq>0):
            for i in range(0,s+1):
                if (freq[i]>0):
                    entropyLeft += 1.0*freq[i]/leftFreq * numpy.log(1.0*freq[i]/leftFreq)
        if (rightFreq>0):
            for i in range(s+1,256):
                if (freq[i]>0):
                    entropyRight += 1.0*freq[i]/rightFreq * numpy.log(1.0*freq[i]/rightFreq)
        entropyRight=-entropyRight; entropyLeft=-entropyLeft
        entropyTotal[s] = entropyRight+entropyLeft
    return numpy.argmax(entropyTotal)
############################################################

############################################################
# BINARY OPENING OPERATION
############################################################
def binary_opening(bImg, iterations=1):
    '''
    Perform binary opening (erosion -> dilation) on a binary image.
    
    Input parameters:
    bImg : (boolean array) Binary input image.
    iterations : (int) Number of iterations for which binary opening
        needs to performed. Default is 1 and the structuring element is
        8-connected 3x3 array of ones.
        
    Returns:
    bImg : (boolean array) Binary image after opening. 
    '''
    for i in range(iterations):
        bImg = ndimage.binary_erosion(bImg)
        bImg = ndimage.binary_dilation(bImg)
    return bImg
############################################################

############################################################
# FIND OUT THE BOUNDARY OF CONNECTED OBJECTS IN A BINARY IMAGE
############################################################
def boundary(bImg):
    '''
    Find the boundary of connected components in a binary image.
    
    Input parameters:
    bImg : (boolean array) Binary input image.
    
    Returns:
    bImgBdry : (boolean array) Binary image with particle boundary
        pixels marked as True. 
    '''
    bImgErode = ndimage.binary_erosion(bImg)
    bImgBdry = (bImg.astype('uint8') - bImgErode.astype('uint8')).astype('bool')
    return bImgBdry
############################################################

#######################################################################
# FIND OUT THE REGION PROPERTIES OF CONNECTED OBJECTS IN A BINARY IMAGE
#######################################################################
def regionProps(bImg,structure=[[1,1,1],[1,1,1],[1,1,1]]):
    '''
    Find region properties of different connected components in a binary
    image. For each label the following are calculated -
    1. bdryPixelList - the list of pixels that form the label boundary.
    2. centroid - the centroid of the label in the (row,col) format.
    3. aspectRatio - the ratio of major axis to the minor axis. A
        value close to 1 means that the label is circular, and a higher
        value means the label is elongated.
    4. minDist - the minimum distance of the label from all other
        labelled components.
        
    Input parameters:
    bImg : (boolean array) Binary input image.
    structure : (list) Connectivity structuring element to use for
        labelling different components in the image.
    
    Returns:
    dictionary : A python dictionary with all the propoerties of
        different labels. 
    '''
    [labelImg,numLabel] = ndimage.label(bImg,structure=structure)
    [row,col] = bImg.shape
    dictionary = {}
    dictionary['bdryPixelList'] = []
    dictionary['centroid'] = []
    dictionary['aspectRatio'] = []
    dictionary['minDist'] = numpy.zeros([numLabel,numLabel])
        
    for i in range(1, numLabel+1):
        bImgLabelN = labelImg==i
        bdry = boundary(bImgLabelN)
        pixelsRC = numpy.nonzero(bdry)
        centerRC = [numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1])]
        major = max(measure.regionprops(bImgLabelN.astype('uint8'))[0]['major_axis_length'],1)
        minor = max(measure.regionprops(bImgLabelN.astype('uint8'))[0]['minor_axis_length'],1)
        AR = 1.0*major/minor
        
        dictionary['bdryPixelList'].append([pixelsRC[0].tolist(),pixelsRC[1].tolist()])
        dictionary['centroid'].append(centerRC)
        dictionary['aspectRatio'].append(AR)
            
    for i in range(numLabel):
        R1,C1 = dictionary['bdryPixelList'][i][0],dictionary['bdryPixelList'][i][1]
        for j in range(numLabel):
            if (i!=j):
                R2,C2 = dictionary['bdryPixelList'][j][0],dictionary['bdryPixelList'][j][1]
                minDist = calculateMinDistance(R1,C1,R2,C2)
                dictionary['minDist'][i][j],dictionary['minDist'][j][i] = minDist,minDist
    return labelImg,numLabel,dictionary
#######################################################################

#######################################################################
# CALCULATE MINIMUM DISTANCE BETWEEN THE BOUNDARY OF TWO OBJECTS
#######################################################################
def calculateMinDistance(R1,C1,R2,C2):
    '''
    Calculate the minimum distance between two set of coordinates in the
    (r,c) format.
    
    Input parameters:
    r1 : (1D list/array) Row coordinates of all the points in object 1.
    c1 : (1D list/array) Col coordinates of all the points in object 1.
    r2 : (1D list/array) Row coordinates of all the points in object 2.
    c2 : (1D list/array) Col coordinates of all the points in object 2.
    
    Returns:
    minDist : (double) Minimum distance between two set of points. 
    '''
    minDist = 1e10
    for r1,c1 in zip(R1,C1):
        for r2,c2 in zip(R2,C2):
            distance = numpy.sqrt((r2-r1)**2 + (c2-c1)**2)
            if (distance<minDist):
                minDist = distance
    return minDist
#######################################################################
