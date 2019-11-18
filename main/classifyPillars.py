import os
import sys
import cv2
import numpy
import pandas
import hyperspy.api as hs
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow import keras
from keras.models import load_model
from sklearn.utils import shuffle
plt.style.use('UMM_ISO9001')

sys.path.append(os.path.abspath('../lib'))
import imageDraw
import imageProcess


####################################################
# FIND OUT IF THE PILLAR IS COLLAPSED OR NOT COLLAPSED USING ML MODEL
####################################################
# excelBook = pandas.ExcelFile('classifyPillars.xlsx')
# sheetName = excelBook.sheet_names[0]
# inputInfo = excelBook.parse(sheetName)

# model = keras.models.load_model('../model/model_01_intermediate_487_accuracy_trainAcc_99.94_testAcc_99.85.h5')
# counter = 0
# outFile = open('classifyPillarsRestoreWater.dat','w')
# outFile.write('InputFile\tTag\tCropStartRow\tCropStartCol\tCropEndRow\tCropEndCol\tClassificationClass\tClassificationClassLabel\n')
# for line in inputInfo.values:
    # inputFile = line[0]
    # rowTopLeft,colTopLeft = line[2],line[1]
    # rowTopRight,colTopRight = line[4],line[3]
    # rowBottomRight,colBottomRight = line[6],line[5]
    # rowBottomLeft,colBottomLeft = line[8],line[7]
    # numPillarsInRow,numPillarsInCol = line[9],line[10]
    # cropSize = int(round(max(2.0*0.75*(colTopRight-colTopLeft)/(numPillarsInRow-1),2.0*0.75*(rowBottomLeft-rowTopLeft)/(numPillarsInCol-1))))
    # if ('.dm3' in inputFile):
        # outputFile = inputFile.replace('.dm3','.png')
    # elif ('.dm4' in inputFile):
        # outputFile = inputFile.replace('.dm4','.png')
    # print('Processing %s' %(inputFile))
    # tag = inputFile.split('/')[-2]
    
    # gImg = imageProcess.readDM4(inputFile)
    # [row,col] = gImg.shape
    # gImgNorm = imageProcess.normalize(gImg,min=30,max=230)
    
    # topRowPillarCentre = numpy.column_stack((\
        # numpy.linspace(rowTopLeft,rowTopRight,numPillarsInRow),\
        # numpy.linspace(colTopLeft,colTopRight,numPillarsInRow)))
    # bottomRowPillarCenter = numpy.column_stack((\
        # numpy.linspace(rowBottomLeft,rowBottomRight,numPillarsInRow),\
        # numpy.linspace(colBottomLeft,colBottomRight,numPillarsInRow)))
        
    # for coordTop,coordBottom in tqdm(zip(topRowPillarCentre,bottomRowPillarCenter)):
        # pillarColumnCoord = numpy.column_stack((numpy.linspace(coordTop[0],coordBottom[0],numPillarsInCol),numpy.linspace(coordTop[1],coordBottom[1],numPillarsInCol)))
        # for r,c in pillarColumnCoord:
            # cropRowStart,cropColStart = int(round(r-cropSize/2)),int(round(c-cropSize/2))
            # cropRowEnd,cropColEnd = int(cropRowStart+cropSize),int(cropColStart+cropSize)
            # if (cropRowStart>=0 and cropColStart>=0 and cropRowEnd<=row and cropColEnd<=col):
                # counter += 1
                # gImgCrop = gImg[cropRowStart:cropRowEnd+1,cropColStart:cropColEnd+1]
                # gImgCrop = cv2.resize(gImgCrop,(32,32),interpolation=cv2.INTER_AREA)
                
                # gImgCrop = gImgCrop.copy().astype('float32')
                # gImgCrop /= 255
                # gImgCrop = gImgCrop.reshape(1,32,32,1)
                # res = model.predict_classes(gImgCrop,batch_size=1)[0]
                # keras.backend.clear_session()
                # if (res==0):
                    # gImgNorm = imageDraw.circle(gImgNorm,(r,c),radius=int(cropSize/4),thickness=8,color=255)
                    # label = 'Collapse'
                # elif (res==1):
                    # gImgNorm = imageDraw.circle(gImgNorm,(r,c),radius=int(cropSize/4),thickness=8,color=0)
                    # label = 'Not collapse'
                # outFile.write('%s\t%s\t%d\t%d\t%d\t%d\t%d\t%s\n' %(inputFile,tag,cropRowStart,cropColStart,cropRowEnd,cropColEnd,res,label))
    # cv2.imwrite(outputFile,gImgNorm)
# outFile.close()
####################################################

####################################################
# PLOT THE DATA
####################################################
numSamples = 100
dataFile = 'classifyPillarsRestoreWater.dat'
df = pandas.read_csv(dataFile,delimiter='\t')
tags = numpy.unique(df['Tag'])
percentStanding = {}
avgList,stdList = [],[]
for tag in tqdm(tags):
    data = df[df['Tag']==tag]['ClassificationClass']
    percentStanding[tag] = []
    for i in range(1000):
        percentStanding[tag].append(100.0*numpy.sum(shuffle(data)[:numSamples])/numSamples)
    avgList.append(numpy.mean(percentStanding[tag]))
    stdList.append(numpy.std(percentStanding[tag]))
    
fig = plt.figure(figsize=(2.8,2.0))
ax = fig.add_axes([0,0,1,1])
ax.fill_between((-1,4.5),y1=avgList[0]-stdList[0],y2=avgList[0]+stdList[0],ec='None',fc='r',alpha=0.25)
ax.axhline(avgList[0],c='r',label='water dry')
ax.fill_between((-1,4.5),y1=avgList[1]-stdList[1],y2=avgList[1]+stdList[1],ec='None',fc='k',alpha=0.25)
ax.axhline(avgList[1],c='k',label='IPA dry')
ax.errorbar(x=[-0.5],y=avgList[2],yerr=stdList[2],c='b',marker='o',ls='None',capsize=2,label='dry box')
ax.errorbar(x=[0,1,2,4],y=avgList[3:],yerr=stdList[3:],c='g',marker='o',ls='None',capsize=2,label='dry box + vacuum')
ax.set_xlim(-1,4.5)
ax.set_ylim(0,100)
ax.set_xticks([0,1,2,3,4])
ax.set_xlabel('t (h)')
ax.set_ylabel('% Standing NP')
ax.legend(numpoints=1)
plt.savefig('percentageStanding.png',format='png')
plt.savefig('percentageStanding.pdf',format='pdf')
plt.close()
####################################################
