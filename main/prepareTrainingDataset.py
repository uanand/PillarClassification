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
firstFrame,lastFrame = 1,4537
zfillVal = 6

#######################################################################
# EVENT HANDLING FUNCTIONS FOR UPDATING DATABSE AND ASSIGNING LABELS
#######################################################################
def updateCounts(val):
    global nopillarCounter,notcollapseCounter,collapseCounter,counter,gImg,outFile
    group = radio.value_selected
    if ("No Pillar" in group):
        nopillarCounter+=1
        targetList.append(2)
        target = 2
    elif ("Not Collapsed" in group):
        notcollapseCounter+=1
        targetList.append(1)
    elif ("Collapsed" in group):
        collapseCounter+=1
        targetList.append(0)
        target = 0
    # imgData = numpy.append(target,gImg.flatten())
    # for i in imgData:
        # outFile.write('%d\t' %(i))
    # outFile.write('\n')
    counter+=1
    if (counter>=len(frameList)):
        plt.close()
        writeClassificationToFile()
        # outFile.close()
    else:
        # gImg = cv2.imread(inputDir+'/'+str(frameList[counter]).zfill(zfillVal)+'.png',0)
        axImage.imshow(gImgStack[:,:,counter],cmap='Greys_r')
        axImage.set_title("C = %d, NC = %d, NP = %d" %(collapseCounter,notcollapseCounter,nopillarCounter))
        axImage.set_xticks([]), axImage.set_yticks([])
    
def loadNewImage(val):
    global counter,gImg
    counter+=1
    if (counter>=len(frameList)):
        plt.close()
        outFile.close()
    else:
        gImg = cv2.imread(inputDir+'/'+str(frameList[counter]).zfill(zfillVal)+'.png',0)
        axImage.imshow(gImg,cmap='Greys_r')
        axImage.set_title("C = %d, NC = %d, NP = %d" %(collapseCounter,notcollapseCounter,nopillarCounter))
        axImage.set_xticks([]), axImage.set_yticks([])
    
def quitProgram(val):
    plt.close()
    outFile.close()
#######################################################################


#######################################################################
# MAIN FUNCTION FOR DISPLAYING IMAGE AND RADIO BUTTONS FOR CLASSIFICATION
#######################################################################
# print("User training for all the cropped images.")
# outFile = open(outputFile,'a')
# collapseCounter,notcollapseCounter,nopillarCounter = 0,0,0
# frameList = range(firstFrame,lastFrame+1); numFrames=len(frameList)
# targetList = numpy.zeros(numFrames,dtype='int32'); targetList[:] = numpy.nan
# # frameList = list(range(firstFrame,lastFrame+1)); random.shuffle(frameList)
# counter = 0
# gImg = cv2.imread(inputDir+'/'+str(frameList[0]).zfill(zfillVal)+'.png',0)
# [row,col],numFrames = gImg.shape,len(frameList)
# gImgStack = numpy.zeros([row,col,numFrames],dtype='uint8')
# for frame,i in zip(frameList,range(len(frameList))):
    # gImgStack[:,:,i] = cv2.imread(inputDir+'/'+str(frame).zfill(zfillVal)+'.png',0)

# fig = plt.figure(figsize=(10,5))
# fig.canvas.set_window_title('Data Labelling Software (Mirsaidov Lab)')
# axImage = fig.add_axes([0.05,0.1,0.4,0.8])
# axRadio = fig.add_axes([0.55,0.3,0.40,0.6])
# axSubmitButton = fig.add_axes([0.55,0.1,0.1,0.1])
# axSkipButton = fig.add_axes([0.7,0.1,0.1,0.1])
# axExitButton = fig.add_axes([0.85,0.1,0.1,0.1])

# axImage.imshow(gImgStack[:,:,0],cmap='Greys_r')
# axImage.set_title("C = %d, NC = %d, NP = %d" %(collapseCounter,notcollapseCounter,nopillarCounter))
# axImage.set_xticks([]), axImage.set_yticks([])
# radio = widgets.RadioButtons(axRadio,('Collapsed','Not Collapsed','No Pillar'))
# submit = widgets.Button(axSubmitButton,'Submit')
# skip = widgets.Button(axSkipButton,'Skip')
# exitSoftware = widgets.Button(axExitButton,'Exit')
# plt.show()

# imgData = submit.on_clicked(updateCounts)
# skip.on_clicked(loadNewImage)
# exitSoftware.on_clicked(quitProgram)
#######################################################################


#######################################################################
# REMOVE DUPLICATES IN THE LABELLED DATASET
#######################################################################
print("Removing duplicates from the labelled dataset.")
labelledDataset = numpy.loadtxt(outputFile,skiprows=0,dtype='int32')
[row,col] = labelledDataset.shape
outFile = open(outputFile,'w')
outFile.write('Label(Collapse=0, Not collapse=1, No pillar=2)\tImage array\n')
for r1 in tqdm(range(row)):
    flag = 0
    data1 = labelledDataset[r1,:]
    for r2 in range(r1+1,row):
        data2 = labelledDataset[r2,:]
        difference = data1-data2
        if (numpy.min(difference)==0 and numpy.max(difference)==0):
            flag = 1
            break
    if (flag==0):
        for i in data1:
            outFile.write('%d\t' %(i))
        outFile.write('\n')
outFile.close()
#######################################################################
