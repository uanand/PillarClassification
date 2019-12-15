import os
import sys
import gc
import cv2
import numpy
import pandas
from tensorflow import keras
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath('../lib'))
import imageDraw
import imageProcess


############################################################
# FIND OUT IF THE PILLAR IS COLLAPSED OR NOT COLLAPSED USING ML MODEL
############################################################
excelFileName = 'classifyPillars.xlsx'
# sheetNameList = ['Plasma+RestoreByWater','RestoreByWater','RestoreBy1.00%HF','RestoreBy0.30%HF','RestoreBy0.10%HF']
sheetNameList = ['RestoreBy0.30%HF']
model = keras.models.load_model('../model/model_01_intermediate_487_accuracy_trainAcc_99.94_testAcc_99.85.h5')

for sheetName in sheetNameList:
    print('Processing %s sheet' %(sheetName))
    df = pandas.read_excel(excelFileName,sheet_name=sheetName,names=['inputFile','colTopLeft','rowTopLeft','colTopRight','rowTopRight','colBottomRight','rowBottomRight','colBottomLeft','rowBottomLeft','numPillarsInRow','numPillarsInCol'],inplace=True)
    
    counter = 0
    outFile = open(sheetName+'.dat','w')
    outFile.write('InputFile\tTag\tPillarID\tCropStartRow\tCropStartCol\tCropEndRow\tCropEndCol\tClassificationClass\tClassificationClassLabel\n')
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
                    
                    gImgCrop = gImgCrop.copy().astype('float32')
                    gImgCrop /= 255
                    gImgCrop = gImgCrop.reshape(1,32,32,1)
                    res = model.predict_classes(gImgCrop,batch_size=1)[0]
                    keras.backend.clear_session()
                    if (res==0):
                        gImgNorm = imageDraw.circle(gImgNorm,(r,c),radius=int(cropSize/4),thickness=8,color=255)
                        label = 'Collapse'
                    elif (res==1):
                        gImgNorm = imageDraw.circle(gImgNorm,(r,c),radius=int(cropSize/4),thickness=8,color=0)
                        label = 'Not collapse'
                    outFile.write('%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%s\n' %(inputFile,tag,pillarID,cropRowStart,cropColStart,cropRowEnd,cropColEnd,res,label))
        cv2.imwrite(outputFile,gImgNorm)
    del df,inputFile,colTopLeft,rowTopLeft,colTopRight,rowTopRight,colBottomRight,rowBottomRight,colBottomLeft,rowBottomLeft,numPillarsInRow,numPillarsInCol,cropSize,outputFile,tag,gImg,gImgNorm,row,col,topRowPillarCentre,bottomRowPillarCenter
    gc.collect()
    outFile.close()
del excelFileName,sheetNameList,model,sheetName
gc.collect()
############################################################
