import os
import sys
import cv2
import numpy
from tqdm import tqdm
from keras.models import load_model

sys.path.append(os.path.abspath('../lib'))
import imageDraw

'''
Manual labelling of dataset is one of the most time consuming part for
machine learning algorithms. We labelled 500 nanopillars as collapsed or
upright and use it to train a model with 95% accuracy. Following this we
used this model to label more nanopillars for preparing training data.
This code generates labelled nanopillars and the ones that are
incorrectly classified were fixed manually to increase the size of
training dataset.
'''

inputDir = '../dataset/allImages'
outputDir = '../dataset/classifiedImages'
labelledDatasetFile = '../dataset/labelledDataset.dat'
outputFile = '../dataset/classifiedDataset.dat'
firstFrame,lastFrame = 1,4537
zfillVal = 6
model = load_model('../model/model2_batchsize_1000_epochs_200.h5')

labelledDataset = numpy.loadtxt(labelledDatasetFile,skiprows=1,dtype='uint8') 
outFile = open(outputFile,'w')
outFile.write('Label(Collapse=0, Not collapse=1)\tImage array\n')
frameList = range(firstFrame,lastFrame+1)
for frame,category in zip(frameList,labelledDataset[:,0]):
    gImgRaw = cv2.imread(inputDir+'/'+str(frame).zfill(zfillVal)+'.png',0)
    
    [row,col] = gImgRaw.shape
    gImg = cv2.resize(gImgRaw,(32,32),interpolation=cv2.INTER_AREA)
    [rowNew,colNew] = gImg.shape
    gImg = gImg.copy().astype('float32')
    gImg /= 255
    gImg = gImg.reshape(1,rowNew,colNew,1)
    
    res = model.predict_classes(gImg,batch_size=1)[0]
    if (res==0):
        label = 'Collapsed'
    elif (res==1):
        label = 'Upright'
        
    if (res!=category):
        print(res,category,frame,label)
        
    outFile.write('%d\t' %(res))
    for i in gImgRaw.flatten():
        outFile.write('%d\t' %(i))
    outFile.write('\n')
    finalImg = imageDraw.labelNextToImage(gImgRaw,label)
    cv2.imwrite(outputDir+'/'+str(frame).zfill(zfillVal)+'.png',finalImg)
outFile.close()
