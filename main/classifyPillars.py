import os
import sys
import cv2
import numpy
import pandas
import hyperspy.api as hs
from tqdm import tqdm
from keras.models import load_model

sys.path.append(os.path.abspath('../lib'))
import imageDraw
import imageProcess

####################################################
# READING THE EXCEL WORKBOOK FOR USER INPUTS
####################################################
excelBook = pandas.ExcelFile('classifyPillars.xlsx')
sheetName = excelBook.sheet_names[0]
inputInfo = excelBook.parse(sheetName)
####################################################


####################################################
model = load_model('../model/model2_batchsize_1000_epochs_200.h5')
counter = 0
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
    outputFile = inputFile.replace('.dm3','.png')
    print('Processing %s' %(inputFile))
    
    f = hs.load(inputFile);
    gImg = f.data
    lowLimit = f.original_metadata['DocumentObjectList']['TagGroup0']['ImageDisplayInfo']['LowLimit']
    highLimit = f.original_metadata['DocumentObjectList']['TagGroup0']['ImageDisplayInfo']['HighLimit']
    gImg[gImg<lowLimit] = lowLimit
    gImg[gImg>highLimit] = highLimit
    gImg = imageProcess.normalize(gImg)
    [row,col] = gImg.shape
    gImgNorm = imageProcess.normalize(gImg,min=30,max=230)
    
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
                counter += 1
                gImgCrop = gImg[cropRowStart:cropRowEnd+1,cropColStart:cropColEnd+1]
                gImgCrop = cv2.resize(gImgCrop,(32,32),interpolation=cv2.INTER_AREA)
                
                gImgCrop = gImgCrop.copy().astype('float32')
                gImgCrop /= 255
                gImgCrop = gImgCrop.reshape(1,32,32,1)
                res = model.predict_classes(gImgCrop,batch_size=1)[0]
                if (res==0):
                    gImgNorm = imageDraw.circle(gImgNorm,(r,c),radius=int(cropSize/4),thickness=8,color=255)
                    # label = 'Collapse'
                elif (res==1):
                    gImgNorm = imageDraw.circle(gImgNorm,(r,c),radius=int(cropSize/4),thickness=8,color=0)
                    # label = 'Not collapse'
    cv2.imwrite(outputFile,gImgNorm)
####################################################
