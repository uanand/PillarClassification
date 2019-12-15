import numpy
import matplotlib.pyplot as plt
plt.style.use('UMM_ISO9001')

inputFile = '/home/utkarsh/Projects/PillarClassification/model/model_01_newData_epochs_500_batchsize_1000_trainAcc_98.33_testAcc_98.33.dat'
fileName = inputFile.replace('.dat','.png')

f = open(inputFile,'r')
text = f.read()
text = text.split("{'loss': [")[1]
loss,text = text.split("], 'accuracy': [")[0],text.split("], 'accuracy': [")[1]
accuracy,text = text.split("], 'val_loss': [")[0],text.split("], 'val_loss': [")[1]
val_loss,text = text.split("], 'val_accuracy': [")[0],text.split("], 'val_accuracy': [")[1]
val_accuracy,text = text.split("]}")[0],text.split("]}")[1]

loss = loss.split(", ")
accuracy = accuracy.split(", ")
val_loss = val_loss.split(", ")
val_accuracy = val_accuracy.split(", ")
epochs = range(1,len(loss)+1)

loss = numpy.asarray(loss,dtype='double')
accuracy = numpy.asarray(accuracy,dtype='double')
val_loss = numpy.asarray(val_loss,dtype='double')
val_accuracy = numpy.asarray(val_accuracy,dtype='double')

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(epochs,loss,label='train')
ax1.plot(epochs,val_loss,label='test')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Loss')
ax1.set_xlim(min(epochs),max(epochs))
ax1.set_ylim(min(min(loss),min(val_loss)),max(max(loss),max(val_loss)))
ax1.legend()
ax2.plot(epochs,accuracy,label='train')
ax2.plot(epochs,val_accuracy,label='test')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Accuracy')
ax2.set_xlim(min(epochs),max(epochs))
ax2.set_ylim(0.95,1)
# ax2.set_ylim(min(min(accuracy),min(val_accuracy)),max(max(accuracy),max(val_accuracy)))
ax2.legend()
plt.savefig(fileName,format='png')
plt.close()
