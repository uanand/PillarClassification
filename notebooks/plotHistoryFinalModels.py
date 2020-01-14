import os,sys
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator

sys.path.append(os.path.abspath('../lib'))
import utils
plt.style.use('UMM_ISO9001')

############################################################
# PLOTTING ACCURACY AND LOSS FOR MODEL 1
############################################################
def plotHistoryModel1():
    history1 = '../model/model_01_test_epochs_200_batchsize_128_trainAcc_85.67_testAcc_96.70.dat'
    history2 = '../model/model_01_test_intermediate_086_epochs_100_batchsize_128_trainAcc_98.74_testAcc_99.45.dat'
    history3 = '../model/model_01_test_intermediate_086_intermediate_091_epochs_100_batchsize_128_trainAcc_98.94_testAcc_99.25.dat'
    
    loss1,accuracy1,val_loss1,val_accuracy1 = utils.parseHistoryDict(history1)
    loss2,accuracy2,val_loss2,val_accuracy2 = utils.parseHistoryDict(history2)
    loss3,accuracy3,val_loss3,val_accuracy3 = utils.parseHistoryDict(history3)
    stopList = [86,91,100]
    learningRateList = [0.0001,0.000001,0.0000001]
    insetLimit = [150,200]
    
    epochs = list(range(1,len(list(range(1,stopList[0]+1))+list(range(1,stopList[1]+1))+list(range(1,stopList[2]+1)))+1))
    loss = numpy.concatenate((loss1[0:stopList[0]],loss2[0:stopList[1]],loss3[0:stopList[2]]))
    val_loss = numpy.concatenate((val_loss1[0:stopList[0]],val_loss2[0:stopList[1]],val_loss3[0:stopList[2]]))
    accuracy = numpy.concatenate((accuracy1[0:stopList[0]],accuracy2[0:stopList[1]],accuracy3[0:stopList[2]]))*100
    val_accuracy = numpy.concatenate((val_accuracy1[0:stopList[0]],val_accuracy2[0:stopList[1]],val_accuracy3[0:stopList[2]]))*100
    
    fig = plt.figure(figsize=(3,2))
    ax = fig.add_axes([0,0,1,1])
    rect1 = patches.Rectangle((1,-0.02),86,1.02,linewidth=None,edgecolor=None,facecolor='#FFFFE0')
    rect2 = patches.Rectangle((86,-0.02),91,1.02,linewidth=None,edgecolor=None,facecolor='#FFEF00')
    rect3 = patches.Rectangle((177,-0.02),100,1.02,linewidth=None,edgecolor=None,facecolor='#EFCC00')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.plot(epochs,loss,label='train',color='b')
    ax.plot(epochs,val_loss,label='test',color='g')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_xlim(min(epochs),max(epochs))
    ax.set_ylim(-0.02,1)
    ax.xaxis.set_minor_locator(MultipleLocator(25))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.legend()
    plt.savefig('model1_loss.png',format='png',transparent=True)
    plt.savefig('model1_loss.pdf',format='pdf',transparent=True)
    plt.close()
    
    fig = plt.figure(figsize=(1,0.66))
    ax = fig.add_axes([0,0,1,1])
    rect2 = patches.Rectangle((insetLimit[0],0),177-insetLimit[0],0.06,linewidth=None,edgecolor=None,facecolor='#FFEF00')
    rect3 = patches.Rectangle((177,0),insetLimit[1]-177,0.06,linewidth=None,edgecolor=None,facecolor='#EFCC00')
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.plot(epochs[insetLimit[0]-1:insetLimit[1]],loss[insetLimit[0]-1:insetLimit[1]],color='b')
    ax.plot(epochs[insetLimit[0]-1:insetLimit[1]],val_loss[insetLimit[0]-1:insetLimit[1]],color='g')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_xlim(insetLimit[0],insetLimit[1])
    ax.set_ylim(0.015,0.055)
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    plt.savefig('model1_loss_inset.png',format='png',transparent=True)
    plt.savefig('model1_loss_inset.pdf',format='pdf',transparent=True)
    plt.close()
    
    fig = plt.figure(figsize=(3,2))
    ax = fig.add_axes([0,0,1,1])
    rect1 = patches.Rectangle((1,50),86,52,linewidth=None,edgecolor=None,facecolor='#FFFFE0')
    rect2 = patches.Rectangle((86,50),91,52,linewidth=None,edgecolor=None,facecolor='#FFEF00')
    rect3 = patches.Rectangle((177,50),100,52,linewidth=None,edgecolor=None,facecolor='#EFCC00')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.plot(epochs,accuracy,label='train',color='b')
    ax.plot(epochs,val_accuracy,label='test',color='g')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlim(min(epochs),max(epochs))
    ax.set_ylim(50,102)
    ax.xaxis.set_minor_locator(MultipleLocator(25))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.legend()
    plt.savefig('model1_accuracy.png',format='png',transparent=True)
    plt.savefig('model1_accuracy.pdf',format='pdf',transparent=True)
    plt.close()
    
    fig = plt.figure(figsize=(1,0.66))
    ax = fig.add_axes([0,0,1,1])
    rect2 = patches.Rectangle((insetLimit[0],98.4),177-insetLimit[0],1.2,linewidth=None,edgecolor=None,facecolor='#FFEF00')
    rect3 = patches.Rectangle((177,98.4),insetLimit[1]-177,1.2,linewidth=None,edgecolor=None,facecolor='#EFCC00')
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.plot(epochs[insetLimit[0]-1:insetLimit[1]],accuracy[insetLimit[0]-1:insetLimit[1]],color='b')
    ax.plot(epochs[insetLimit[0]-1:insetLimit[1]],val_accuracy[insetLimit[0]-1:insetLimit[1]],color='g')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlim(insetLimit[0],insetLimit[1])
    ax.set_ylim(98.4,99.6)
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    plt.savefig('model1_accuracy_inset.png',format='png',transparent=True)
    plt.savefig('model1_accuracy_inset.pdf',format='pdf',transparent=True)
    plt.close()
############################################################

############################################################
# PLOTTING ACCURACY AND LOSS FOR MODEL 2
############################################################
def plotHistoryModel2():
    history1 = '../model/model_02_20200106_epochs_200_batchsize_128_trainAcc_51.79_testAcc_53.85.dat'
    history2 = '../model/model_02_20200106_intermediate_025_epochs_20_batchsize_128_trainAcc_99.61_testAcc_99.80.dat'
    history3 = '../model/model_02_20200106_intermediate_025_intermediate_007_epochs_20_batchsize_128_trainAcc_99.57_testAcc_99.76.dat'
    
    loss1,accuracy1,val_loss1,val_accuracy1 = utils.parseHistoryDict(history1)
    loss2,accuracy2,val_loss2,val_accuracy2 = utils.parseHistoryDict(history2)
    loss3,accuracy3,val_loss3,val_accuracy3 = utils.parseHistoryDict(history3)
    stopList = [25,7,20]
    learningRateList = [0.01,0.001,0.0001]
    insetLimit = [25,40]
    
    epochs = list(range(1,len(list(range(1,stopList[0]+1))+list(range(1,stopList[1]+1))+list(range(1,stopList[2]+1)))+1))
    loss = numpy.concatenate((loss1[0:stopList[0]],loss2[0:stopList[1]],loss3[0:stopList[2]]))
    val_loss = numpy.concatenate((val_loss1[0:stopList[0]],val_loss2[0:stopList[1]],val_loss3[0:stopList[2]]))
    accuracy = numpy.concatenate((accuracy1[0:stopList[0]],accuracy2[0:stopList[1]],accuracy3[0:stopList[2]]))*100
    val_accuracy = numpy.concatenate((val_accuracy1[0:stopList[0]],val_accuracy2[0:stopList[1]],val_accuracy3[0:stopList[2]]))*100
    
    fig = plt.figure(figsize=(3,2))
    ax = fig.add_axes([0,0,1,1])
    rect1 = patches.Rectangle((1,-0.02),25,0.52,linewidth=None,edgecolor=None,facecolor='#FFFFE0')
    rect2 = patches.Rectangle((25,-0.02),7,0.52,linewidth=None,edgecolor=None,facecolor='#FFEF00')
    rect3 = patches.Rectangle((32,-0.02),20,0.52,linewidth=None,edgecolor=None,facecolor='#EFCC00')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.plot(epochs,loss,label='train',color='b')
    ax.plot(epochs,val_loss,label='test',color='g')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_xlim(min(epochs),max(epochs))
    ax.set_ylim(-0.02,0.5)
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.legend()
    plt.savefig('model2_loss.png',format='png',transparent=True)
    plt.savefig('model2_loss.pdf',format='pdf',transparent=True)
    plt.close()
    
    fig = plt.figure(figsize=(1,0.66))
    ax = fig.add_axes([0,0,1,1])
    rect2 = patches.Rectangle((insetLimit[0],0.004),32-insetLimit[0],0.022,linewidth=None,edgecolor=None,facecolor='#FFEF00')
    rect3 = patches.Rectangle((32,0.004),insetLimit[1]-32,0.022,linewidth=None,edgecolor=None,facecolor='#EFCC00')
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.plot(epochs[insetLimit[0]-1:insetLimit[1]],loss[insetLimit[0]-1:insetLimit[1]],color='b')
    ax.plot(epochs[insetLimit[0]-1:insetLimit[1]],val_loss[insetLimit[0]-1:insetLimit[1]],color='g')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_xlim(insetLimit[0],insetLimit[1])
    ax.set_ylim(0.004,0.026)
    ax.xaxis.set_minor_locator(MultipleLocator(2.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.005))
    plt.savefig('model2_loss_inset.png',format='png',transparent=True)
    plt.savefig('model2_loss_inset.pdf',format='pdf',transparent=True)
    plt.close()
    
    fig = plt.figure(figsize=(3,2))
    ax = fig.add_axes([0,0,1,1])
    rect1 = patches.Rectangle((1,70),25,31,linewidth=None,edgecolor=None,facecolor='#FFFFE0')
    rect2 = patches.Rectangle((25,70),7,31,linewidth=None,edgecolor=None,facecolor='#FFEF00')
    rect3 = patches.Rectangle((32,70),20,31,linewidth=None,edgecolor=None,facecolor='#EFCC00')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.plot(epochs,accuracy,label='train',color='b')
    ax.plot(epochs,val_accuracy,label='test',color='g')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlim(min(epochs),max(epochs))
    ax.set_ylim(70,101)
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(2.5))
    ax.legend()
    plt.savefig('model2_accuracy.png',format='png',transparent=True)
    plt.savefig('model2_accuracy.pdf',format='pdf',transparent=True)
    plt.close()
    
    fig = plt.figure(figsize=(1,0.66))
    ax = fig.add_axes([0,0,1,1])
    rect2 = patches.Rectangle((insetLimit[0],99.25),32-insetLimit[0],0.6,linewidth=None,edgecolor=None,facecolor='#FFEF00')
    rect3 = patches.Rectangle((32,99.25),insetLimit[1]-32,0.6,linewidth=None,edgecolor=None,facecolor='#EFCC00')
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.plot(epochs[insetLimit[0]-1:insetLimit[1]],accuracy[insetLimit[0]-1:insetLimit[1]],color='b')
    ax.plot(epochs[insetLimit[0]-1:insetLimit[1]],val_accuracy[insetLimit[0]-1:insetLimit[1]],color='g')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlim(insetLimit[0],insetLimit[1])
    ax.set_ylim(99.25,99.85)
    ax.xaxis.set_minor_locator(MultipleLocator(2.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.125))
    plt.savefig('model2_accuracy_inset.png',format='png',transparent=True)
    plt.savefig('model2_accuracy_inset.pdf',format='pdf',transparent=True)
    plt.close()
############################################################

############################################################
# PLOTTING ACCURACY AND LOSS FOR MODEL VGG
############################################################
def plotHistoryModelVGG():
    history1 = '../model/vgg16_20200107_epochs_20_batchsize_128_trainAcc_52.39_testAcc_52.27.dat'
    history2 = '../model/vgg16_20200107_intermediate_003_epochs_20_batchsize_128_trainAcc_99.61_testAcc_99.56.dat'
    history3 = '../model/vgg16_20200107_intermediate_003_intermediate_020_epochs_50_batchsize_128_trainAcc_99.81_testAcc_99.76.dat'
    history4 = '../model/vgg16_20200107_intermediate_003_intermediate_020_intermediate_030_epochs_100_batchsize_128_trainAcc_99.91_testAcc_99.93.dat'
    
    loss1,accuracy1,val_loss1,val_accuracy1 = utils.parseHistoryDict(history1)
    loss2,accuracy2,val_loss2,val_accuracy2 = utils.parseHistoryDict(history2)
    loss3,accuracy3,val_loss3,val_accuracy3 = utils.parseHistoryDict(history3)
    loss4,accuracy4,val_loss4,val_accuracy4 = utils.parseHistoryDict(history4)
    stopList = [3,20,30,100]
    learningRateList = [0.01,0.001,0.001,0.0001]
    insetLimit = [120,153]
    
    epochs = list(range(1,len(list(range(1,stopList[0]+1))+list(range(1,stopList[1]+1))+list(range(1,stopList[2]+1))+list(range(1,stopList[3]+1)))+1))
    loss = numpy.concatenate((loss1[0:stopList[0]],loss2[0:stopList[1]],loss3[0:stopList[2]],loss4[0:stopList[3]]))
    val_loss = numpy.concatenate((val_loss1[0:stopList[0]],val_loss2[0:stopList[1]],val_loss3[0:stopList[2]],val_loss4[0:stopList[3]]))
    accuracy = numpy.concatenate((accuracy1[0:stopList[0]],accuracy2[0:stopList[1]],accuracy3[0:stopList[2]],accuracy4[0:stopList[3]]))*100
    val_accuracy = numpy.concatenate((val_accuracy1[0:stopList[0]],val_accuracy2[0:stopList[1]],val_accuracy3[0:stopList[2]],val_accuracy4[0:stopList[3]]))*100
    
    fig = plt.figure(figsize=(3,2))
    ax = fig.add_axes([0,0,1,1])
    rect1 = patches.Rectangle((1,-0.02),3,1.02,linewidth=None,edgecolor=None,facecolor='#FFFFE0')
    rect2 = patches.Rectangle((4,-0.02),50,1.02,linewidth=None,edgecolor=None,facecolor='#FFEF00')
    rect3 = patches.Rectangle((54,-0.02),100,1.02,linewidth=None,edgecolor=None,facecolor='#EFCC00')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.plot(epochs,loss,label='train',color='b')
    ax.plot(epochs,val_loss,label='test',color='g')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_xlim(min(epochs),max(epochs))
    ax.set_ylim(-0.02,1)
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.legend()
    plt.savefig('modelVGG_loss.png',format='png',transparent=True)
    plt.savefig('modelVGG_loss.pdf',format='pdf',transparent=True)
    plt.close()
    
    fig = plt.figure(figsize=(1,0.66))
    ax = fig.add_axes([0,0,1,1])
    rect3 = patches.Rectangle((120,0.002),insetLimit[1]-120,0.006,linewidth=None,edgecolor=None,facecolor='#EFCC00')
    ax.add_patch(rect3)
    ax.plot(epochs[insetLimit[0]-1:insetLimit[1]],loss[insetLimit[0]-1:insetLimit[1]],color='b')
    ax.plot(epochs[insetLimit[0]-1:insetLimit[1]],val_loss[insetLimit[0]-1:insetLimit[1]],color='g')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_xlim(insetLimit[0],insetLimit[1])
    ax.set_ylim(0.002,0.008)
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(0.00125))
    plt.savefig('modelVGG_loss_inset.png',format='png',transparent=True)
    plt.savefig('modelVGG_loss_inset.pdf',format='pdf',transparent=True)
    plt.close()
    
    fig = plt.figure(figsize=(3,2))
    ax = fig.add_axes([0,0,1,1])
    rect1 = patches.Rectangle((1,80),3,21,linewidth=None,edgecolor=None,facecolor='#FFFFE0')
    rect2 = patches.Rectangle((4,80),50,21,linewidth=None,edgecolor=None,facecolor='#FFEF00')
    rect3 = patches.Rectangle((54,80),100,21,linewidth=None,edgecolor=None,facecolor='#EFCC00')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.plot(epochs,accuracy,label='train',color='b')
    ax.plot(epochs,val_accuracy,label='test',color='g')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlim(min(epochs),max(epochs))
    ax.set_ylim(80,101)
    ax.set_yticks([80,85,90,95,100])
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(2.5))
    ax.legend()
    plt.savefig('modelVGG_accuracy.png',format='png',transparent=True)
    plt.savefig('modelVGG_accuracy.pdf',format='pdf',transparent=True)
    plt.close()
    
    fig = plt.figure(figsize=(1,0.66))
    ax = fig.add_axes([0,0,1,1])
    rect3 = patches.Rectangle((120,99.6),insetLimit[1]-120,0.4,linewidth=None,edgecolor=None,facecolor='#EFCC00')
    ax.add_patch(rect3)
    ax.plot(epochs[insetLimit[0]-1:insetLimit[1]],accuracy[insetLimit[0]-1:insetLimit[1]],color='b')
    ax.plot(epochs[insetLimit[0]-1:insetLimit[1]],val_accuracy[insetLimit[0]-1:insetLimit[1]],color='g')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlim(insetLimit[0],insetLimit[1])
    ax.set_ylim(99.6,100)
    ax.set_yticks([99.6,99.8,100.0])
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.savefig('modelVGG_accuracy_inset.png',format='png',transparent=True)
    plt.savefig('modelVGG_accuracy_inset.pdf',format='pdf',transparent=True)
    plt.close()
############################################################


plotHistoryModel1()
plotHistoryModel2()
plotHistoryModelVGG()


