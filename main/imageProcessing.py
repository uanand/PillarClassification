import os
import sys
import numpy
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage

sys.path.append(os.path.abspath('../lib'))
import loadData
import imageProcess

'''
Image processing method to classify nanopillars as collapsed or upright.
The following process flow is used to classify nanopillars - 

1. Load and invert the image. The pixels in the nanopillars appear
    brighter and the background appears darker.
2. Apply Otsu/Kapur/Mean threshold to binarize the image. After this,
    the nanopillars are typically marked as Tru and the background as
    False.
3. Perform morphological operations to clean the binary image and label
    the connected components.
4. Identify the central object. Typically this the the nanopillar we
    want to classify.
5. Calculate the distance of the nearest connected component (NND) from
    the central object.
6. Calculate the aspect ratio (AR) of the central object.
7. If AR > 1.5 or NND < 16, mark the nanopillar as collapsed.
8. Mark the nnopillar as upright otherwise.
'''

############################################################
# LOAD THE LABELLED DATASET AND SPLIT INTO TRAINING AND TEST
############################################################
row,col = 64,64
xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd = loadData.loadPillarData(fileName='../dataset/labelledDataset.dat',numClasses=2,row=row,col=col,rotFlag=True,flipFlag=True,RGB=False,shuffleFlag=True)
print (xTrain.shape,yTrainInd.shape,xTest.shape,yTestInd.shape,numpy.sum(yTrainInd,axis=0),numpy.sum(yTestInd,axis=0))
############################################################

############################################################
# IMAGE PROCESSING METHOD TO SEGMENT AND CLASSIFY NANOPILLARS
############################################################
outFile = open('imageProcessingLabel.dat','w')
outFile.write('Actual label\tPredicted label\n')

# RUN ON TRAINING DATA
counter1,counter2 = 0,0
totalCount,incorrectCount = 0,0
for frame,x,y in tqdm(zip(range(yTrain.size),xTrain,yTrain)):
    totalCount+=1
    gImg = numpy.reshape(x,(row,col))
    gImgNorm = imageProcess.normalize(gImg)
    gImg = imageProcess.invert(gImgNorm)
    
    # THRESHOLD METHOD 1 (OTSU)
    bImg = gImg >= imageProcess.otsuThreshold(gImg)
    
    # THRESHOLD METHOD 2 (KAPUR)
    # bImg = gImg >= imageProcess.threshold_kapur(gImg)
    
    # THRESHOLD METHOD 2 (MEAN)
    # bImg = gImg >= numpy.mean(gImg)
    
    bImg = imageProcess.binary_opening(bImg,iterations=4)
    labelImg,numLabel,dictionary = imageProcess.regionProps(bImg)
    
    distanceFromCentre = 1e10
    for i in range(numLabel):
        rCenter,cCenter = dictionary['centroid'][i]
        distance = numpy.sqrt((rCenter-row/2.0)**2 + (cCenter-col/2.0)**2)
        if (distance < distanceFromCentre):
            distanceFromCentre = distance
            centerLabel = i+1
    nearestNeighborDistance = 1e10
    for i in range(numLabel):
        if (i!=centerLabel-1):
            if (dictionary['minDist'][centerLabel-1][i]<nearestNeighborDistance):
                nearestNeighborDistance = dictionary['minDist'][centerLabel-1][i]
    AR = dictionary['aspectRatio'][centerLabel-1]
    
    if (AR>1.5 or nearestNeighborDistance<16):
        predictedLabel = 'Collapsed'
        yHat = 0
    else:
        predictedLabel = 'Not collapsed'
        yHat = 1
    if (y==0):
        actualLabel = 'Collapsed'
    else:
        actualLabel = 'Not collapsed'
    if (actualLabel!=predictedLabel):
        incorrectCount+=1
        
    if (actualLabel==predictedLabel and counter1<10):
        cv2.imwrite('correct_'+str(frame+1)+'_gImg.png',gImgNorm)
        cv2.imwrite('correct_'+str(frame+1)+'_bImg.png',bImg*255)
        counter1+=1
    if (actualLabel!=predictedLabel and counter2<10):
        cv2.imwrite('incorrect_'+str(frame+1)+'_gImg.png',gImgNorm)
        cv2.imwrite('incorrect_'+str(frame+1)+'_bImg.png',bImg*255)
        counter2+=1
    outFile.write('%s\t%s\n' %(actualLabel,predictedLabel))
    
# RUN ON TEST DATA
for i,x,y in tqdm(zip(range(yTest.size),xTest,yTest)):
    totalCount+=1
    gImg = numpy.reshape(x,(row,col))
    gImgNorm = imageProcess.normalize(gImg)
    gImg = imageProcess.invert(gImgNorm)
    
    # THRESHOLD METHOD 1 (OTSU)
    # bImg = gImg >= imageProcess.otsuThreshold(gImg)
    
    # THRESHOLD METHOD 2 (MEAN)
    bImg = gImg >= numpy.mean(gImg)
    bImg = imageProcess.binary_opening(bImg,iterations=4)
    
    labelImg,numLabel,dictionary = imageProcess.regionProps(bImg)
    
    distanceFromCentre = 1e10
    for i in range(numLabel):
        rCenter,cCenter = dictionary['centroid'][i]
        distance = numpy.sqrt((rCenter-row/2.0)**2 + (cCenter-col/2.0)**2)
        if (distance < distanceFromCentre):
            distanceFromCentre =distance
            centerLabel = i+1
    nearestNeighborDistance = 1e10
    for i in range(numLabel):
        if (i!=centerLabel-1):
            if (dictionary['minDist'][centerLabel-1][i]<nearestNeighborDistance):
                nearestNeighborDistance = dictionary['minDist'][centerLabel-1][i]
    AR = dictionary['aspectRatio'][centerLabel-1]
    
    if (AR>1.5 or nearestNeighborDistance<16):
        predictedLabel = 'Collapsed'
        yHat = 0
    else:
        predictedLabel = 'Not collapsed'
        yHat = 1
    if (y==0):
        actualLabel = 'Collapsed'
    else:
        actualLabel = 'Not collapsed'
    if (actualLabel!=predictedLabel):
        incorrectCount+=1
    outFile.write('%s\t%s\n' %(actualLabel,predictedLabel))
outFile.close()
print ('Accuracy = %f' %(100.0*(totalCount-incorrectCount)/totalCount))
############################################################
