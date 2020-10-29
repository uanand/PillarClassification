'''
Preparing cropped images for labelling. The raw DM3/DM4 files are
entered as a list in an excel sheet ('prepareCropImages.xlsx') and small
images centered around nanopillars are cropped and saved in a user
defined output directory.
'''

import os
import sys
import numpy
import pandas
import cv2
import hyperspy.api as hs
import matplotlib.pyplot as plt
from skimage import transform
from tqdm import tqdm

sys.path.append(os.path.abspath('../lib'))
import imageProcess
import imageDraw

outputDir = '../dataset/allImages'

####################################################
# READING THE EXCEL WORKBOOK FOR USER INPUTS
####################################################
excelBook = pandas.ExcelFile('prepareCropImages.xlsx')
sheetName = excelBook.sheet_names[0]
inputInfo = excelBook.parse(sheetName)
####################################################

####################################################
counter = 1
zfillVal = 6
for line in inputInfo.values:
    inputFile = line[0]
    rowTopLeft = line[1]
    colTopLeft = line[2]
    rowTopRight = line[3]
    colTopRight = line[4]
    rowBottomLeft = line[5]
    colBottomLeft = line[6]
    numPillarsInRow = line[7]
    numPillarsInCol = line[8]
    cropSize = int(round(max(2.0*0.75*(colTopRight-colTopLeft)/(numPillarsInRow-1),2.0*0.75*(rowBottomLeft-rowTopLeft)/(numPillarsInCol-1))))
    
    rowBottomRight = rowBottomLeft-(rowTopLeft-rowTopRight)
    colBottomRight = colBottomLeft-(colTopLeft-colTopRight)
    
    f = hs.load(inputFile);
    gImg = f.data
    lowLimit = f.original_metadata['DocumentObjectList']['TagGroup0']['ImageDisplayInfo']['LowLimit']
    highLimit = f.original_metadata['DocumentObjectList']['TagGroup0']['ImageDisplayInfo']['HighLimit']
    gImg[gImg<lowLimit] = lowLimit
    gImg[gImg>highLimit] = highLimit
    gImg = imageProcess.normalize(gImg)
    [row,col] = gImg.shape
    
    topRowPillarCentre = numpy.column_stack((\
        numpy.linspace(rowTopLeft,rowTopRight,numPillarsInRow),\
        numpy.linspace(colTopLeft,colTopRight,numPillarsInRow)))
    bottomRowPillarCenter = numpy.column_stack((\
        numpy.linspace(rowBottomLeft,rowBottomRight,numPillarsInRow),\
        numpy.linspace(colBottomLeft,colBottomRight,numPillarsInRow)))
        
    for coordTop,coordBottom in tqdm(zip(topRowPillarCentre,bottomRowPillarCenter)):
        pillarColumnCoord = numpy.column_stack((numpy.linspace(coordTop[0],coordBottom[0],numPillarsInCol),numpy.linspace(coordTop[1],coordBottom[1],numPillarsInCol)))
        for r,c in pillarColumnCoord:
            cropRowStart,cropColStart = int(round(r-cropSize/2)),int(round(c-cropSize/2))
            cropRowEnd,cropColEnd = int(cropRowStart+cropSize),int(cropColStart+cropSize)
            if (cropRowStart>=0 and cropColStart>=0 and cropRowEnd<=row and cropColEnd<=col):
                gImgCrop = gImg[cropRowStart:cropRowEnd+1,cropColStart:cropColEnd+1]
                gImgCrop = cv2.resize(gImgCrop,(64,64),interpolation=cv2.INTER_AREA)
                cv2.imwrite(outputDir+'/'+str(counter).zfill(zfillVal)+'.png',gImgCrop)
                counter += 1
####################################################
