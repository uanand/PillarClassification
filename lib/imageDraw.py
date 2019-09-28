import cv2

#######################################################################
# DRAW A CIRCLE IN AN IMAGE
#######################################################################
def circle(img,center,radius,thickness=1,color=255):
    [row,col] = img.shape
    img = cv2.circle(img,center=(int(center[1]),int(center[0])),radius=radius,color=color,thickness=thickness)
    return img
#######################################################################
