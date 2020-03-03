import os
import sys
import cv2
import numpy
from skimage.transform import rotate
from sklearn.utils import shuffle

sys.path.append(os.path.abspath('../lib'))
import imageProcess


data = numpy.loadtxt('../dataset/labelledDataset.dat',skiprows=1)
data = shuffle(data,random_state=10)
x,y = data[:,1:],data[:,0]
[row,col] = x.shape

imgSize = 64
numRowCollage,numColCollage = 3,20
counter_0,counter_1 = 0,0
img_0 = numpy.zeros([imgSize*numRowCollage,imgSize*numColCollage],dtype='uint8')
img_1 = numpy.zeros([imgSize*numRowCollage,imgSize*numColCollage],dtype='uint8')
for i in range(row):
    if (y[i]==0 and counter_0<numRowCollage*numColCollage):
        row_0,col_0 = int(int(counter_0/numColCollage)*imgSize),int((counter_0%numColCollage)*imgSize)
        counter_0+=1
        img = numpy.reshape(x[i,:],(64,64))
        img = imageProcess.normalize(img)
        img_0[row_0:row_0+imgSize,col_0:col_0+imgSize] = img
    elif (y[i]==1 and counter_1<numRowCollage*numColCollage):
        row_1,col_1 = int(int(counter_1/numColCollage)*imgSize),int((counter_1%numColCollage)*imgSize)
        counter_1+=1
        img = numpy.reshape(x[i,:],(64,64))
        img = imageProcess.normalize(img)
        img_1[row_1:row_1+imgSize,col_1:col_1+imgSize] = img
    if (counter_0>=numRowCollage*numColCollage and counter_1>=numRowCollage*numColCollage):
        break
cv2.imwrite('collage_collapsed.png',img_1)
cv2.imwrite('collage_not_collapsed.png',img_0)

###############################
# ROTATE AND CONCATENATE IMAGES
# gImg = cv2.imread('/home/utkarsh/Projects/PillarClassification/dataset/allImages/002475.png',0)
# gImg_90 = (rotate(gImg,90)*255).astype('uint8')
# gImg_180 = (rotate(gImg,180)*255).astype('uint8')
# gImg_270 = (rotate(gImg,270)*255).astype('uint8')
# finalImg = numpy.column_stack((gImg,gImg_90,gImg_180,gImg_270))
# cv2.imwrite('image.png',finalImg)
