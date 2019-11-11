import cv2
import numpy
import random
from skimage.transform import rotate

# inputDir = '../dataset/allImages'
# firstFrame,lastFrame = 1,1762
# zfillVal = 6
# frameList = list(range(firstFrame,lastFrame+1))
# random.shuffle(frameList)

# counter = 0
# for frame1 in range(8):
    # for frame2 in range(16):
        # gImg = cv2.imread(inputDir+'/'+str(frameList[counter]).zfill(zfillVal)+'.png',0)
        # counter+=1
        # try:
            # columnImg = numpy.column_stack((columnImg,gImg))
        # except:
            # columnImg = gImg.copy()
    # try:
        # rowImg = numpy.row_stack((rowImg,columnImg))
    # except:
        # rowImg = columnImg.copy()
    # del columnImg
# cv2.imwrite('collage.png',rowImg)


###############################
# ROTATE AND CONCATENATE IMAGES
gImg = cv2.imread('/home/utkarsh/Projects/PillarClassification/dataset/allImages/002475.png',0)
gImg_90 = (rotate(gImg,90)*255).astype('uint8')
gImg_180 = (rotate(gImg,180)*255).astype('uint8')
gImg_270 = (rotate(gImg,270)*255).astype('uint8')
finalImg = numpy.column_stack((gImg,gImg_90,gImg_180,gImg_270))
cv2.imwrite('image.png',finalImg)


# utils.imagesForLabelDataset(fileName='../dataset/labelledDataset.dat',numClasses=2,dirList=['/home/utkarsh/Projects/PillarClassification/dataset/labelledImages/Collapse','/home/utkarsh/Projects/PillarClassification/dataset/labelledImages/Not collapse'])
