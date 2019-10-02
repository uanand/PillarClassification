import os
import sys
import cv2
import numpy
from tqdm import tqdm
from keras.models import load_model

sys.path.append(os.path.abspath('../lib'))
import imageDraw

inputDir = '../dataset/allImages'
outputDir = '../dataset/classifiedImages'
firstFrame,lastFrame = 1,4537
zfillVal = 6
model = load_model('../model/model1_batchsize_50_epochs_500.h5')

frameList = range(firstFrame,lastFrame+1)
for frame in tqdm(frameList):
    gImgRaw = cv2.imread(inputDir+'/'+str(frame).zfill(zfillVal)+'.png',0)
    [row,col] = gImgRaw.shape
    gImg = gImgRaw.copy().astype('float32')
    gImg /= 255
    gImg = gImg.reshape(1,row,col,1)
    
    res = model.predict_classes(gImg,batch_size=1)[0]
    if (res==0):
        label = 'Collapse'
    elif (res==1):
        label = 'Not collapse'
        
    finalImg = imageDraw.labelNextToImage(gImgRaw,label)
    cv2.imwrite(outputDir+'/'+str(frame).zfill(zfillVal)+'.png',finalImg)
