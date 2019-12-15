import os
import sys
import numpy
import cv2
import random
from matplotlib import widgets
import matplotlib.pyplot as plt
from tqdm import tqdm

inputDir = '../dataset/allImages'
outputFile = '../dataset/labelledDataset.dat'
classificationFile = '../dataset/classifiedDataset.dat'
firstFrame,lastFrame = 1,4537
zfillVal = 6
method = 'useClassificationResult'#'makeNewDataset'
# frameList = range(firstFrame,lastFrame+1)
# numFrames=len(frameList)

############################################################
# EVENT HANDLING FUNCTIONS FOR UPDATING DATABSE AND ASSIGNING LABELS
############################################################
def updateLabel(val):
    global targetList,targetNameList,counter
    group = radio.value_selected
    if (group=='No Pillar'):
        targetList[counter] = 2
        targetNameList[counter] = 'No pillar'
    elif (group=='Not Collapsed'):
        targetList[counter] = 1
        targetNameList[counter] = 'Not collapse'
    elif (group=='Collapsed'):
        targetList[counter] = 0
        targetNameList[counter] = 'Collapse'
    counter+=1
    if (counter>=numFrames):
        plt.close()
        writeClassificationToFile(gImgStack,targetList,counter,outputFile)
    else:
        axImage.imshow(gImgStack[:,:,counter],cmap='Greys_r')
        axImage.set_title(targetNameList[counter])
        axImage.set_xlabel('Frame %d/%d' %(counter+1,numFrames))
    
def loadNewImage(val):
    global counter
    counter+=1
    if (counter<numFrames):
        axImage.imshow(gImgStack[:,:,counter],cmap='Greys_r')
        axImage.set_title(targetNameList[counter])
        axImage.set_xlabel('Frame %d/%d' %(counter+1,numFrames))
    else:
        plt.close()
        writeClassificationToFile(gImgStack,targetList,counter,outputFile)
    
def quitProgram(val):
    plt.close()
    writeClassificationToFile(gImgStack,targetList,counter,outputFile)
    
def press(event):
    if (event.key == ' '):
        loadNewImage('foo')
############################################################


############################################################
# WRITING THE CLASSIFICATION LABELS TO FILE
############################################################
def writeClassificationToFile(gImgStack,targetList,counter,outputFile):
    outFile = open(outputFile,'w')
    outFile.write('Label(Collapse=0, Not collapse=1, No pillar=2)\tImage array\n')
    for i in range(counter):
        outFile.write('%d\t' %(targetList[i]))
        for pixel in gImgStack[:,:,i].flatten():
            outFile.write('%d\t' %(pixel))
        outFile.write('\n')
    outFile.close()
    removeDuplicates(outputFile)
############################################################


############################################################
# REMOVE DUPLICATES FROM THE TRAINING DATASET 
############################################################
def removeDuplicates(outputFile):
    print("Removing duplicates from the labelled dataset.")
    try:
        labelledDataset = numpy.loadtxt(outputFile,skiprows=1,dtype='int32')
        targetList = labelledDataset[:,0]
        gImgList = labelledDataset[:,1:]
        [row,col] = gImgList.shape
        outFile = open(outputFile,'w')
        outFile.write('Label(Collapse=0, Not collapse=1, No pillar=2)\tImage array\n')
        for r1 in tqdm(range(row)):
            flag = 0
            gImg1 = gImgList[r1,:]
            for r2 in range(r1+1,row):
                gImg2 = gImgList[r2,:]
                difference = gImg1-gImg2
                if (numpy.min(difference)==0 and numpy.max(difference)==0):
                    flag = 1
                    break
            if (flag==0):
                outFile.write('%d\t' %(targetList[r1]))
                for i in gImg1:
                    outFile.write('%d\t' %(i))
                outFile.write('\n')
        outFile.close()
    except:
        pass
############################################################
    

############################################################
# MAIN FUNCTION FOR DISPLAYING IMAGE AND RADIO BUTTONS FOR CLASSIFICATION
############################################################
print("User training for all the cropped images.")
if (method=='makeNewDataset'):
    outFile = open(outputFile,'a')
    collapseCounter,notcollapseCounter,nopillarCounter = 0,0,0
    targetList = numpy.zeros(numFrames,dtype='int32')
    targetList[:] = numpy.nan
    counter = 0
    gImg = cv2.imread(inputDir+'/'+str(frameList[0]).zfill(zfillVal)+'.png',0)
    [row,col],numFrames = gImg.shape,len(frameList)
    gImgStack = numpy.zeros([row,col,numFrames],dtype='uint8')
    for frame,i in zip(frameList,range(len(frameList))):
        gImgStack[:,:,i] = cv2.imread(inputDir+'/'+str(frame).zfill(zfillVal)+'.png',0)
    xlabel = ''
    
elif (method=='useClassificationResult'):
    outFile = open(outputFile,'w')
    data = numpy.loadtxt(classificationFile,skiprows=1,dtype='uint8')
    targetList = data[:,0]
    gImg = data[:,1:]
    [numFrames,numPixels] = gImg.shape
    row,col = int(numpy.sqrt(numPixels)),int(numpy.sqrt(numPixels))
    gImgStack = numpy.zeros([row,col,numFrames],dtype='uint8')
    counter = 0
    for frame in range(numFrames):
        gImgStack[:,:,frame] = numpy.reshape(gImg[frame,:],(row,col))
    collapseCounter = numpy.sum(targetList==0)
    notcollapseCounter = numpy.sum(targetList==1)
    nopillarCounter = numpy.sum(targetList==2)
    targetNameList = []
    for target in targetList:
        if (target==0):
            targetNameList.append('Collapse')
        elif (target==1):
            targetNameList.append('Not collapse')
        elif (target==2):
            targetNameList.append('No pillar')
    
fig = plt.figure(figsize=(6,3))
fig.canvas.set_window_title('Data Labelling Software (Mirsaidov Lab)')
fig.canvas.mpl_connect('key_press_event',press)
axImage = fig.add_axes([0.05,0.1,0.4,0.8])
axRadio = fig.add_axes([0.55,0.3,0.40,0.6])
axSubmitButton = fig.add_axes([0.55,0.1,0.1,0.1])
axSkipButton = fig.add_axes([0.7,0.1,0.1,0.1])
axExitButton = fig.add_axes([0.85,0.1,0.1,0.1])

axImage.imshow(gImgStack[:,:,0],cmap='Greys_r')
if (method=='useClassificationResult'):
    axImage.set_title(targetNameList[0])
    axImage.set_xlabel('Frame %d/%d' %(counter+1,numFrames))
elif (method=='makeNewDataset'):
    axImage.set_title("C = %d, NC = %d, NP = %d" %(collapseCounter,notcollapseCounter,nopillarCounter))
axImage.set_xticks([]), axImage.set_yticks([])
radio = widgets.RadioButtons(axRadio,('Collapse','Not collapse','No pillar'))
submit = widgets.Button(axSubmitButton,'Submit')
skip = widgets.Button(axSkipButton,'Skip')
exitSoftware = widgets.Button(axExitButton,'Exit')
plt.show()

submit.on_clicked(updateLabel)
skip.on_clicked(loadNewImage)
exitSoftware.on_clicked(quitProgram)
############################################################
