import os
import sys
import numpy
import cv2
from matplotlib import widgets
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath('../lib'))
import imageProcess
import imageDraw

inputDir = '../dataset/allImages'
firstFrame,lastFrame = 1,1830
zfillVal = 6

#######################################################################
# 
#######################################################################
def updateCounts(value):
    if (value=='Collapsed'):
        cv2.imwrite('../dataset/training/collapse/'+str(counterCollapsed).zfill(zfillVal)+'png',gImg)
        counterCollapse += 1
    elif (value=='Not Collapsed'):
        cv2.imwrite('../dataset/training/notcollapse/'+str(counterNotCollapsed).zfill(zfillVal)+'png',gImg)
        counterNotCollapse += 1
    elif (value=='No Pillar'):
        cv2.imwrite('../dataset/training/nopillar/'+str(counterNoPillar).zfill(zfillVal)+'png',gImg)
        counterNoPillar += 1
#######################################################################

collapseCounter,notcollapseCounter,nopillarCounter = 0,0,0
for frame in [1]:#range(firstFrame,lastFrame+1):
    gImg = cv2.imread(inputDir+'/'+str(frame).zfill(zfillVal)+'.png',0)
    
    fig = plt.figure(figsize=(10,5))
    axImage = fig.add_axes([0.1,0.1,0.4,0.8])
    axRadio = fig.add_axes([0.55,0.3,0.35,0.6])
    # axCounter = fig.add_axes([0.75,0.3,0.15,0.6])
    axSubmitButton = fig.add_axes([0.55,0.1,0.15,0.1])
    axSkipButton = fig.add_axes([0.75,0.1,0.15,0.1])
    
    axImage.imshow(gImg)
    radio = widgets.RadioButtons(axRadio,('Collapsed, Count = %d' %(collapseCounter),'Not Collapsed, Count = %d' %(notcollapseCounter),'No Pillar, Count = %d' %(nopillarCounter)))
    submit = widgets.Button(axSubmitButton,'Submit')
    skip = widgets.Button(axSkipButton,'Skip')
    # radio.on_clicked(assignGroup)
    submit.on_clicked(updateCounts)
    skip.on_clicked(loadNewImage)
    plt.show()
