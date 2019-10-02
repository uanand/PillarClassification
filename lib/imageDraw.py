import cv2
import numpy
import matplotlib.pyplot as plt

############################################################
# DRAW A CIRCLE IN AN IMAGE
############################################################
def circle(img,center,radius,thickness=1,color=255):
    [row,col] = img.shape
    img = cv2.circle(img,center=(int(center[1]),int(center[0])),radius=radius,color=color,thickness=thickness)
    return img
############################################################

############################################################
# WRITE CLASSIFICATION LABEL ALONGSIDE IMAGE
############################################################
def labelNextToImage(gImg,label):
    [row,col] = gImg.shape
    labelImg = numpy.zeros([row,col],dtype='uint8')
    position = (36,1) # row,col format
    fontScale = 0.3
    color = 255
    thickness = 1
    cv2.putText(labelImg,label,(position[1],position[0]),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=fontScale,color=color,thickness=thickness,bottomLeftOrigin=False)
    finalImg = numpy.column_stack((gImg,labelImg))
    return finalImg
############################################################
