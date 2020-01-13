import os,sys
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append(os.path.abspath('../lib'))
import utils
plt.style.use('UMM_ISO9001')

############################################################
# PLOTTING ACCURACY AND LOSS FOR MODEL 1
############################################################
history1 = '../model/model_01_test_epochs_200_batchsize_128_trainAcc_85.67_testAcc_96.70.dat'
history2 = '../model/model_01_test_intermediate_086_epochs_100_batchsize_128_trainAcc_98.74_testAcc_99.45.dat'
history3 = '../model/model_01_test_intermediate_086_intermediate_091_epochs_100_batchsize_128_trainAcc_98.94_testAcc_99.25.dat'

loss1,accuracy1,val_loss1,val_accuracy1 = utils.parseHistoryDict(history1)
loss2,accuracy2,val_loss2,val_accuracy2 = utils.parseHistoryDict(history2)
loss3,accuracy3,val_loss3,val_accuracy3 = utils.parseHistoryDict(history3)
stopList = [86,91,100]
learningRateList = [0.0001,0.000001,0.0000001]
epochs = list(range(1,len(list(range(1,stopList[0]+1))+list(range(1,stopList[1]+1))+list(range(1,stopList[2]+1)))+1))
epochs1,epochs2,epochs3 = epochs[0:stopList[0]],epochs[stopList[0]:stopList[0]+stopList[1]],epochs[stopList[0]+stopList[1]:stopList[0]+stopList[1]+stopList[2]]

fig = plt.figure(figsize=(3,2))
ax = fig.add_axes([0,0,1,1])
rect1 = patches.Rectangle((1,-0.02),86,1.02,linewidth=None,edgecolor=None,facecolor='#FFFFE0')
rect2 = patches.Rectangle((86,-0.02),91,1.02,linewidth=None,edgecolor=None,facecolor='#FFEF00')
rect3 = patches.Rectangle((177,-0.02),100,1.02,linewidth=None,edgecolor=None,facecolor='#EFCC00')
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.plot(epochs1,loss1[:stopList[0]],label='train',color='b')
ax.plot(epochs1,val_loss1[:stopList[0]],label='test',color='g')
ax.plot([epochs1[-1]]+epochs2,numpy.concatenate(([loss1[stopList[0]-1]],loss2[:stopList[1]])),color='b')
ax.plot([epochs1[-1]]+epochs2,numpy.concatenate(([val_loss1[stopList[0]-1]],val_loss2[:stopList[1]])),color='g')
ax.plot([epochs2[-1]]+epochs3,numpy.concatenate(([loss2[stopList[1]-1]],loss3[:stopList[2]])),color='b')
ax.plot([epochs2[-1]]+epochs3,numpy.concatenate(([val_loss2[stopList[1]-1]],val_loss3[:stopList[2]])),color='g')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_xlim(min(epochs),max(epochs))
ax.set_ylim(-0.02,1)
ax.legend()
plt.savefig('model1_loss.png',format='png')
plt.savefig('model1_loss.pdf',format='pdf')
plt.close()

fig = plt.figure(figsize=(1.5,1))
ax = fig.add_axes([0,0,1,1])
rect1 = patches.Rectangle((1,0),86,0.06,linewidth=None,edgecolor=None,facecolor='#FFFFE0')
rect2 = patches.Rectangle((86,0),91,0.06,linewidth=None,edgecolor=None,facecolor='#FFEF00')
rect3 = patches.Rectangle((177,0),100,0.06,linewidth=None,edgecolor=None,facecolor='#EFCC00')
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.plot(epochs1,loss1[:stopList[0]],label='train',color='b')
ax.plot(epochs1,val_loss1[:stopList[0]],label='test',color='g')
ax.plot([epochs1[-1]]+epochs2,numpy.concatenate(([loss1[stopList[0]-1]],loss2[:stopList[1]])),color='b')
ax.plot([epochs1[-1]]+epochs2,numpy.concatenate(([val_loss1[stopList[0]-1]],val_loss2[:stopList[1]])),color='g')
ax.plot([epochs2[-1]]+epochs3,numpy.concatenate(([loss2[stopList[1]-1]],loss3[:stopList[2]])),color='b')
ax.plot([epochs2[-1]]+epochs3,numpy.concatenate(([val_loss2[stopList[1]-1]],val_loss3[:stopList[2]])),color='g')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_xlim(150,200)
ax.set_ylim(0.00,0.06)
plt.savefig('model1_loss_inset.png',format='png')
plt.savefig('model1_loss_inset.pdf',format='pdf')
plt.close()

fig = plt.figure(figsize=(3,2))
ax = fig.add_axes([0,0,1,1])
rect1 = patches.Rectangle((1,0.5),86,0.97,linewidth=None,edgecolor=None,facecolor='#FFFFE0')
rect2 = patches.Rectangle((86,0.5),91,0.97,linewidth=None,edgecolor=None,facecolor='#FFEF00')
rect3 = patches.Rectangle((177,0.5),100,0.97,linewidth=None,edgecolor=None,facecolor='#EFCC00')
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.plot(epochs1,accuracy1[:stopList[0]],label='train',color='b')
ax.plot(epochs1,val_accuracy1[:stopList[0]],label='test',color='g')
ax.plot([epochs1[-1]]+epochs2,numpy.concatenate(([accuracy1[stopList[0]-1]],accuracy2[:stopList[1]])),color='b')
ax.plot([epochs1[-1]]+epochs2,numpy.concatenate(([val_accuracy1[stopList[0]-1]],val_accuracy2[:stopList[1]])),color='g')
ax.plot([epochs2[-1]]+epochs3,numpy.concatenate(([accuracy2[stopList[1]-1]],accuracy3[:stopList[2]])),color='b')
ax.plot([epochs2[-1]]+epochs3,numpy.concatenate(([val_accuracy2[stopList[1]-1]],val_accuracy3[:stopList[2]])),color='g')
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy (%)')
ax.set_xlim(min(epochs),max(epochs))
ax.set_ylim(0.5,1.02)
ax.legend()
plt.savefig('model1_accuracy.png',format='png')
plt.savefig('model1_accuracy.pdf',format='pdf')
plt.close()
############################################################
