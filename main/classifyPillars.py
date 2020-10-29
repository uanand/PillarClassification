'''
Classify the all the nanopillars from the raw DM3/DM4 files as collapsed
or upright. The list of images are entered in the excel file in the
correpsonsing sheets, and each nanopillar is classified using three
models (DNN, CNN, and VGG). The classification label, and the classification
time taken for each nanopillar is written to a text file.
'''

import os
import sys
import gc
import cv2
import numpy
import pandas
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import load_model
from time import time

sys.path.append(os.path.abspath('../lib'))
import imageDraw
import imageProcess


############################################################
# FIND OUT IF THE PILLAR IS COLLAPSED OR NOT COLLAPSED USING
# NEURAL NETWORK MODEL
############################################################

###### NAME OF THE INPUT EXCEL FILE AND SHEETS
excelFileName = 'classifyPillars.xlsx'
sheetNameList = ['Original','Plasma+RestoreByWater','RestoreByWater','RestoreBy1.00%HF','RestoreBy0.30%HF','RestoreBy0.10%HF']

###### LOAD THREE DIFFERENT TYPES OF MODELS FOR CLASSIFYING EACH NANOPILLAR
model_DNN = keras.models.load_model('../model/model_01_test_intermediate_086_intermediate_091_accuracy_trainAcc_99.42_testAcc_99.47.h5')
model_CNN = keras.models.load_model('../model/model_02_20200106_intermediate_025_intermediate_007_accuracy_trainAcc_99.84_testAcc_99.83.h5')
model_VGG = keras.models.load_model('../model/vgg16_20200107_intermediate_003_intermediate_020_intermediate_030_intermediate_099_accuracy_trainAcc_99.93_testAcc_99.93.h5')

###### ITERATE THROUGH EACH SHEET
for sheetName in sheetNameList:
    print('Processing %s sheet' %(sheetName))
    
    ###### READ THE EXCEL SHEET TO EXTRACT DM4 FILE DETAILS
    df = pandas.read_excel(excelFileName,sheet_name=sheetName,names=['inputFile','colTopLeft','rowTopLeft','colTopRight','rowTopRight','colBottomRight','rowBottomRight','colBottomLeft','rowBottomLeft','numPillarsInRow','numPillarsInCol'],inplace=True)
    
    ###### CREATING EMPTY FILES FOR SAVING OUTPUT FROM DNN, CNN, AND VGG MODELS 
    counter = 0
    outFile_DNN = open(sheetName+'_DNN.dat','w')
    outFile_CNN = open(sheetName+'_CNN.dat','w')
    outFile_VGG = open(sheetName+'_VGG.dat','w')
    outFile_ALL = open(sheetName+'_ALL.dat','w')
    outFile_DNN.write('InputFile\tTag\tPillarID\tCropStartRow\tCropStartCol\tCropEndRow\tCropEndCol\tClassificationClass\tClassificationClassLabel\tTime(s)\n')
    outFile_CNN.write('InputFile\tTag\tPillarID\tCropStartRow\tCropStartCol\tCropEndRow\tCropEndCol\tClassificationClass\tClassificationClassLabel\tTime(s)\n')
    outFile_VGG.write('InputFile\tTag\tPillarID\tCropStartRow\tCropStartCol\tCropEndRow\tCropEndCol\tClassificationClass\tClassificationClassLabel\tTime(s)\n')
    outFile_ALL.write('InputFile\tTag\tPillarID\tCropStartRow\tCropStartCol\tCropEndRow\tCropEndCol\tClassificationClass\tClassificationClassLabel\tTime(s)\n')
    
    ###### ITERATING THROUGH EACH DM3/DM4 IMAGE
    for inputFile,colTopLeft,rowTopLeft,colTopRight,rowTopRight,colBottomRight,rowBottomRight,colBottomLeft,rowBottomLeft,numPillarsInRow,numPillarsInCol in df.values:
        cropSize = int(round(max(2.0*0.75*(colTopRight-colTopLeft)/(numPillarsInRow-1),2.0*0.75*(rowBottomLeft-rowTopLeft)/(numPillarsInCol-1))))
        if ('.dm3' in inputFile):
            outputFile = inputFile.replace('.dm3','.png')
        elif ('.dm4' in inputFile):
            outputFile = inputFile.replace('.dm4','.png')
        print('Processing %s' %(inputFile))
        tag = inputFile.split('/')[-2]
        
        gImg = imageProcess.readDM4(inputFile)
        [row,col] = gImg.shape
        gImgNorm = imageProcess.normalize(gImg,min=30,max=230)
        
        topRowPillarCentre = numpy.column_stack((\
            numpy.linspace(rowTopLeft,rowTopRight,numPillarsInRow),\
            numpy.linspace(colTopLeft,colTopRight,numPillarsInRow)))
        bottomRowPillarCenter = numpy.column_stack((\
            numpy.linspace(rowBottomLeft,rowBottomRight,numPillarsInRow),\
            numpy.linspace(colBottomLeft,colBottomRight,numPillarsInRow)))
            
        ###### ITERATING THROUGH EACH NANOPILLAR IN THE INPUT IMAGE
        pillarID = 0
        for coordTop,coordBottom in zip(topRowPillarCentre,bottomRowPillarCenter):
            pillarColumnCoord = numpy.column_stack((numpy.linspace(coordTop[0],coordBottom[0],numPillarsInCol),numpy.linspace(coordTop[1],coordBottom[1],numPillarsInCol)))
            for r,c in pillarColumnCoord:
                cropRowStart,cropColStart = int(round(r-cropSize/2)),int(round(c-cropSize/2))
                cropRowEnd,cropColEnd = int(cropRowStart+cropSize),int(cropColStart+cropSize)
                if (cropRowStart>=0 and cropColStart>=0 and cropRowEnd<=row and cropColEnd<=col):
                    pillarID += 1
                    counter += 1
                    gImgCrop = gImg[cropRowStart:cropRowEnd+1,cropColStart:cropColEnd+1]
                    gImgCrop = cv2.resize(gImgCrop,(32,32),interpolation=cv2.INTER_AREA)
                    gImgCrop = imageProcess.normalize(gImgCrop)
                    gImgCropRGB = imageProcess.convetToRGB(gImgCrop)
                    
                    gImgCrop = gImgCrop.copy().astype('float32'); gImgCrop /= 255; gImgCrop = gImgCrop.reshape(1,32,32,1)
                    gImgCropRGB = gImgCropRGB.copy().astype('float32'); gImgCropRGB /= 255; gImgCropRGB = gImgCropRGB.reshape(1,32,32,3)
                    
                    ###### CLASSIFICATION FOR A CROPPED NANOPILLAR IMAGE
                    tic_DNN = time(); res_DNN = model_DNN.predict_classes(gImgCrop,batch_size=1)[0]; toc_DNN = time()-tic_DNN;
                    tic_CNN = time(); res_CNN = model_CNN.predict_classes(gImgCrop,batch_size=1)[0]; toc_CNN = time()-tic_CNN;
                    tic_VGG = time(); res_VGG = numpy.argmax(model_VGG.predict(gImgCropRGB,batch_size=1)); toc_VGG = time()-tic_VGG;
                    res_ALL = min(res_DNN,res_CNN,res_VGG); toc_ALL = toc_DNN+toc_CNN+toc_VGG;
                    keras.backend.clear_session()
                    if (res_ALL==0):
                        gImgNorm = imageDraw.circle(gImgNorm,(r,c),radius=int(cropSize/4),thickness=8,color=255)
                        label = 'Collapse'
                    elif (res_ALL==1):
                        gImgNorm = imageDraw.circle(gImgNorm,(r,c),radius=int(cropSize/4),thickness=8,color=0)
                        label = 'Not collapse'
                    outFile_DNN.write('%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%f\n' %(inputFile,tag,pillarID,cropRowStart,cropColStart,cropRowEnd,cropColEnd,res_DNN,label,toc_DNN))
                    outFile_CNN.write('%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%f\n' %(inputFile,tag,pillarID,cropRowStart,cropColStart,cropRowEnd,cropColEnd,res_CNN,label,toc_CNN))
                    outFile_VGG.write('%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%f\n' %(inputFile,tag,pillarID,cropRowStart,cropColStart,cropRowEnd,cropColEnd,res_VGG,label,toc_VGG))
                    outFile_ALL.write('%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%f\n' %(inputFile,tag,pillarID,cropRowStart,cropColStart,cropRowEnd,cropColEnd,res_ALL,label,toc_ALL))
        cv2.imwrite(outputFile,gImgNorm)
    del df,inputFile,colTopLeft,rowTopLeft,colTopRight,rowTopRight,colBottomRight,rowBottomRight,colBottomLeft,rowBottomLeft,numPillarsInRow,numPillarsInCol,cropSize,outputFile,tag,gImg,gImgNorm,row,col,topRowPillarCentre,bottomRowPillarCenter
    gc.collect()
    outFile_DNN.close()
    outFile_CNN.close()
    outFile_VGG.close()
    outFile_ALL.close()
del excelFileName,sheetNameList,model_DNN,model_CNN,model_VGG,sheetName
gc.collect()
############################################################
