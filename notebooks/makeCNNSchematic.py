import cv2
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.style.use('UMM_ISO9001')

############################################################
# FUNCTION TO MAKE THE SURFACES OF A CUBOID USING THE
# CENTRE AND SIZE IN EACH DIMENSION
############################################################
def makeVerticesCuboid(x0,y0,z0,l,w,h):
    x,y,z = x0-l/2.0,y0-w/2.0,z0-h/2.0
    A = [x,y,z]
    B = [x,y+w-1,z]
    C = [x+l-1,y+w-1,z]
    D = [x+l-1,y,z]
    E = [x,y,z+h-1]
    F = [x,y+w-1,z+h-1]
    G = [x+l-1,y+w-1,z+h-1]
    H = [x+l-1,y,z+h-1]
    
    vertices = [\
                [A,B,C,D],\
                [E,F,G,H],\
                [A,B,F,E],\
                [C,D,H,G],\
                [A,D,H,E,],\
                [B,C,G,F]]
    return vertices
############################################################


############################################################
# PREPROCESS AND USER INPUTS
############################################################
inputImg = cv2.imread('000214.png',0)
[row,col] = inputImg.shape
X,Y = numpy.arange(row),numpy.arange(col)
XX,YY = numpy.meshgrid(X,Y)
XX = XX-row/2.0
YY = YY-col/2.0
convolutionColor = '#D79837'
maxPoolColor = '#CD5C5C'
denseLayerColor = '#00BFFF'
softmaxColor = '#808080'
flattenColor = '#000000'
############################################################


############################################################
# SCHEMATIC FOR DNN MODEL
############################################################
fig = plt.figure(figsize=(5,5))
ax = fig.add_axes([0,0,1,1],projection='3d')

ax.plot_surface(XX,YY,numpy.zeros_like(inputImg),facecolors=plt.cm.gray(inputImg),shade=False)

# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,30,70,5,5),facecolors='w',linewidths=0.75,edgecolors=flattenColor))
# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,50,50,5,5),facecolors='w',linewidths=0.75,edgecolors=denseLayerColor))
# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,70,50,5,5),facecolors='w',linewidths=0.75,edgecolors=denseLayerColor))
# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,90,30,5,5),facecolors='w',linewidths=0.75,edgecolors=denseLayerColor))
# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,110,10,5,5),facecolors='w',linewidths=0.75,edgecolors=softmaxColor))

# ax.set_xlim(-70,70)
# ax.set_ylim(-70,70)
# ax.set_zlim(-10,130)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.grid(False)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# plt.savefig('schematic_DNN.png',format='png')
# plt.savefig('schematic_DNN.pdf',format='pdf')
plt.show()
############################################################


############################################################
# SCHEMATIC FOR 1 LAYER CNN MODEL
############################################################

############################################################


############################################################
# SCHEMATIC FOR VGG MODEL
############################################################
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_axes([0,0,1,1],projection="3d")

# ax.plot_surface(XX,YY,numpy.zeros_like(inputImg),facecolors=plt.cm.gray(inputImg),shade=False)

# ####### RESPECTIVE FACECOLOR WITH BLACK EDGES
# # ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,40,32,32,5),facecolors=convolutionColor,linewidths=0.75,edgecolors='k'))
# # ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,50,32,32,5),facecolors=convolutionColor,linewidths=0.75,edgecolors='k'))

# # ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,70,16,16,5),facecolors=maxPoolColor,linewidths=0.75,edgecolors='k'))
# # ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,80,16,16,5),facecolors=convolutionColor,linewidths=0.75,edgecolors='k'))
# # ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,90,16,16,5),facecolors=convolutionColor,linewidths=0.75,edgecolors='k'))

# # ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,110,8,8,5),facecolors=maxPoolColor,linewidths=0.75,edgecolors='k'))
# # ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,120,8,8,5),facecolors=convolutionColor,linewidths=0.75,edgecolors='k'))
# # ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,130,8,8,5),facecolors=convolutionColor,linewidths=0.75,edgecolors='k'))
# # ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,140,8,8,5),facecolors=convolutionColor,linewidths=0.75,edgecolors='k'))

# # ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,160,4,4,5),facecolors=maxPoolColor,linewidths=0.75,edgecolors='k'))
# # ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,170,4,4,5),facecolors=convolutionColor,linewidths=0.75,edgecolors='k'))
# # ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,180,4,4,5),facecolors=convolutionColor,linewidths=0.75,edgecolors='k'))
# # ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,190,4,4,5),facecolors=convolutionColor,linewidths=0.75,edgecolors='k'))

# # ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,210,2,2,5),facecolors=maxPoolColor,linewidths=0.75,edgecolors='k'))
# # ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,220,2,2,5),facecolors=convolutionColor,linewidths=0.75,edgecolors='k'))
# # ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,230,2,2,5),facecolors=convolutionColor,linewidths=0.75,edgecolors='k'))
# # ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,240,2,2,5),facecolors=convolutionColor,linewidths=0.75,edgecolors='k'))

# # ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,260,1,1,5),facecolors=maxPoolColor,linewidths=0.75,edgecolors='k'))
# # ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,270,1,1,20),facecolors=denseLayerColor,linewidths=0.75,edgecolors='k'))
# # ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,295,1,1,10),facecolors=softmaxColor,linewidths=0.75,edgecolors='k'))

# ####### RESPECTIVE EDGECOLOR WITH WHITE FACES
# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,40,32,32,5),facecolors='w',linewidths=0.75,edgecolors=convolutionColor))
# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,50,32,32,5),facecolors='w',linewidths=0.75,edgecolors=convolutionColor))

# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,70,16,16,5),facecolors='w',linewidths=0.75,edgecolors=maxPoolColor))
# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,80,16,16,5),facecolors='w',linewidths=0.75,edgecolors=convolutionColor))
# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,90,16,16,5),facecolors='w',linewidths=0.75,edgecolors=convolutionColor))

# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,110,8,8,5),facecolors='w',linewidths=0.75,edgecolors=maxPoolColor))
# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,120,8,8,5),facecolors='w',linewidths=0.75,edgecolors=convolutionColor))
# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,130,8,8,5),facecolors='w',linewidths=0.75,edgecolors=convolutionColor))
# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,140,8,8,5),facecolors='w',linewidths=0.75,edgecolors=convolutionColor))

# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,160,4,4,5),facecolors='w',linewidths=0.75,edgecolors=maxPoolColor))
# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,170,4,4,5),facecolors='w',linewidths=0.75,edgecolors=convolutionColor))
# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,180,4,4,5),facecolors='w',linewidths=0.75,edgecolors=convolutionColor))
# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,190,4,4,5),facecolors='w',linewidths=0.75,edgecolors=convolutionColor))

# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,210,2,2,5),facecolors='w',linewidths=0.75,edgecolors=maxPoolColor))
# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,220,2,2,5),facecolors='w',linewidths=0.75,edgecolors=convolutionColor))
# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,230,2,2,5),facecolors='w',linewidths=0.75,edgecolors=convolutionColor))
# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,240,2,2,5),facecolors='w',linewidths=0.75,edgecolors=convolutionColor))

# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,260,1,1,5),facecolors='w',linewidths=0.75,edgecolors=maxPoolColor))
# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,270,1,1,20),facecolors='w',linewidths=0.75,edgecolors=denseLayerColor))
# ax.add_collection3d(Poly3DCollection(makeVerticesCuboid(0,0,295,1,1,10),facecolors='w',linewidths=0.75,edgecolors=softmaxColor))

# ax.set_xlim(-20,20)
# ax.set_ylim(-20,20)
# ax.set_zlim(-5,300)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.savefig('schematic_VGG.png',format='png')
# plt.show()
############################################################
