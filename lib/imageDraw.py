import cv2
import numpy

############################################################
# DRAW A CIRCLE IN AN IMAGE
############################################################
def circle(img,center,radius,thickness=1,color=255):
    '''
    Draw circle on an image.
    
    Input parameters:
    img : (uint8 2D array) Image numpy array in grayscale
    centre : (tuple/list) Centre coordinates of the circle as a tuple
        (row,column) or a list [row,column].
    radius : (int) Radius of the circle in pixels.
    thickness : (int) Thickness of the circle boundary. If 1, then a 1
        pixel thick circle is drawn on the image. If set to a negative
        value, a filled circle is drawn.  
    color : (int) Grayscale intensity of the circle. 0 makes a black
        circle, 255 makes a white circle. For an intemediate grayscale
        value choose a number between 0 and 255.
        
    Returns:
    img : (uint8 2D array) Image numpy array with circle drawn based on
        the input parameters.
    '''
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
