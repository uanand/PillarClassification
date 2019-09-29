import os
import sys
import numpy
import cv2
import random
from matplotlib import widgets
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath('../lib'))
import imageProcess
import imageDraw

inputDir = '../dataset/allImages'
firstFrame,lastFrame = 1,1762
zfillVal = 6

#######################################################################
# 
#######################################################################
def updateCounts(val):
    global nopillarCounter,notcollapseCounter,collapseCounter,counter,gImg,outFile
    group = radio.value_selected
    if ("No Pillar" in group):
        nopillarCounter+=1
        target = 2
    elif ("Not Collapsed" in group):
        notcollapseCounter+=1
        target = 1
    elif ("Collapsed" in group):
        collapseCounter+=1
        target = 0
    imgData = numpy.append(target,gImg.flatten())
    for i in imgData:
        outFile.write('%d\t' %(i))
    outFile.write('\n')
    counter+=1
    if (counter>=len(frameList)):
        plt.close()
        outFile.close()
    else:
        gImg = cv2.imread(inputDir+'/'+str(frameList[counter]).zfill(zfillVal)+'.png',0)
        axImage.imshow(gImg,cmap='Greys_r')
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

try:
    outFile = open('../dataset/training/labelledData.dat','r+')
except:
    outFile = open('../dataset/training/labelledData.dat','w')
collapseCounter,notcollapseCounter,nopillarCounter = 0,0,0
frameList = list(range(firstFrame,lastFrame+1)); random.shuffle(frameList)
counter = 0
gImg = cv2.imread(inputDir+'/'+str(frameList[counter]).zfill(zfillVal)+'.png',0)

fig = plt.figure(figsize=(10,5))
fig.canvas.set_window_title('Data Labelling Software (Mirsaidov Lab)')
axImage = fig.add_axes([0.05,0.1,0.4,0.8])
axRadio = fig.add_axes([0.55,0.3,0.40,0.6])
axSubmitButton = fig.add_axes([0.55,0.1,0.1,0.1])
axSkipButton = fig.add_axes([0.7,0.1,0.1,0.1])
axExitButton = fig.add_axes([0.85,0.1,0.1,0.1])

axImage.imshow(gImg,cmap='Greys_r')
axImage.set_title("C = %d, NC = %d, NP = %d" %(collapseCounter,notcollapseCounter,nopillarCounter))
axImage.set_xticks([]), axImage.set_yticks([])
radio = widgets.RadioButtons(axRadio,('Collapsed','Not Collapsed','No Pillar'))
submit = widgets.Button(axSubmitButton,'Submit')
skip = widgets.Button(axSkipButton,'Skip')
exitSoftware = widgets.Button(axExitButton,'Exit')
plt.show()

imgData = submit.on_clicked(updateCounts)
skip.on_clicked(loadNewImage)
exitSoftware.on_clicked(quitProgram)
