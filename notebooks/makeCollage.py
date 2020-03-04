import os
import sys
import cv2
import numpy
from skimage.transform import rotate
from sklearn.utils import shuffle

sys.path.append(os.path.abspath('../lib'))
import imageProcess

############################################################
# READING INPUT FILE
############################################################
data = numpy.loadtxt('../dataset/labelledDataset.dat',skiprows=1)
data = shuffle(data,random_state=4)
imgSize = 64
x,y = data[:,1:],data[:,0]
[row,col] = x.shape
############################################################


############################################################
# MAKE COLLAGE FOR THE COLLAPSED AND NOT COLLAPSED PILLARS
############################################################
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
############################################################


############################################################
# GENERATE IMAGES FOR DATA AUGMENTATION
############################################################
numRowCollage,numColCollage = 3,8
counter_0,counter_1 = 0,0
img_0 = numpy.zeros([imgSize*numRowCollage,imgSize*numColCollage],dtype='uint8')
img_1 = numpy.zeros([imgSize*numRowCollage,imgSize*numColCollage],dtype='uint8')
for i in range(row):
    if (y[i]==0 and counter_0<numRowCollage):
        row_0,col_0 = int(counter_0*imgSize),0
        counter_0+=1
        print (row_0,col_0,counter_0)
        
        img = numpy.reshape(x[i,:],(64,64))
        img = imageProcess.normalize(img)
        img_0[row_0:row_0+imgSize,col_0:col_0+imgSize] = img
        
        img_090 = (rotate(img,90) *255).astype('uint8')
        img_180 = (rotate(img,180)*255).astype('uint8')
        img_270 = (rotate(img,270)*255).astype('uint8')
        col_0 = 1*imgSize; img_0[row_0:row_0+imgSize,col_0:col_0+imgSize] = img_090
        col_0 = 2*imgSize; img_0[row_0:row_0+imgSize,col_0:col_0+imgSize] = img_180
        col_0 = 3*imgSize; img_0[row_0:row_0+imgSize,col_0:col_0+imgSize] = img_270
        
        img_flipx = numpy.fliplr(img)
        img_090_flipx = numpy.fliplr(img_090)
        img_180_flipx = numpy.fliplr(img_180)
        img_270_flipx = numpy.fliplr(img_270)
        col_0 = 4*imgSize; img_0[row_0:row_0+imgSize,col_0:col_0+imgSize] = img_flipx
        col_0 = 5*imgSize; img_0[row_0:row_0+imgSize,col_0:col_0+imgSize] = img_090_flipx
        col_0 = 6*imgSize; img_0[row_0:row_0+imgSize,col_0:col_0+imgSize] = img_180_flipx
        col_0 = 7*imgSize; img_0[row_0:row_0+imgSize,col_0:col_0+imgSize] = img_270_flipx
        
    elif (y[i]==1 and counter_1<numRowCollage):
        row_1,col_1 = int(counter_1*imgSize),0
        counter_1+=1
        
        img = numpy.reshape(x[i,:],(64,64))
        img = imageProcess.normalize(img)
        img_1[row_1:row_1+imgSize,col_1:col_1+imgSize] = img
        
        img_090 = (rotate(img,90) *255).astype('uint8')
        img_180 = (rotate(img,180)*255).astype('uint8')
        img_270 = (rotate(img,270)*255).astype('uint8')
        col_1 = 1*imgSize; img_1[row_1:row_1+imgSize,col_1:col_1+imgSize] = img_090
        col_1 = 2*imgSize; img_1[row_1:row_1+imgSize,col_1:col_1+imgSize] = img_180
        col_1 = 3*imgSize; img_1[row_1:row_1+imgSize,col_1:col_1+imgSize] = img_270
        
        img_flipx = numpy.fliplr(img)
        img_090_flipx = numpy.fliplr(img_090)
        img_180_flipx = numpy.fliplr(img_180)
        img_270_flipx = numpy.fliplr(img_270)
        col_1 = 4*imgSize; img_1[row_1:row_1+imgSize,col_1:col_1+imgSize] = img_flipx
        col_1 = 5*imgSize; img_1[row_1:row_1+imgSize,col_1:col_1+imgSize] = img_090_flipx
        col_1 = 6*imgSize; img_1[row_1:row_1+imgSize,col_1:col_1+imgSize] = img_180_flipx
        col_1 = 7*imgSize; img_1[row_1:row_1+imgSize,col_1:col_1+imgSize] = img_270_flipx
        
    if (counter_0>=numRowCollage and counter_1>=numRowCollage):
        break
cv2.imwrite('collage_collapsed_augmentation.png',img_1)
cv2.imwrite('collage_not_collapsed_augmentation.png',img_0)
############################################################
