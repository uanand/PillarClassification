import cv2
import numpy
import random

inputDir = '../dataset/allImages'
firstFrame,lastFrame = 1,1762
zfillVal = 6
frameList = list(range(firstFrame,lastFrame+1))
random.shuffle(frameList)

counter = 0
for frame1 in range(8):
    for frame2 in range(16):
        gImg = cv2.imread(inputDir+'/'+str(frameList[counter]).zfill(zfillVal)+'.png',0)
        counter+=1
        try:
            columnImg = numpy.column_stack((columnImg,gImg))
        except:
            columnImg = gImg.copy()
    try:
        rowImg = numpy.row_stack((rowImg,columnImg))
    except:
        rowImg = columnImg.copy()
    del columnImg
cv2.imwrite('collage.png',rowImg)