import numpy
import matplotlib.pyplot as plt

inputFile = '/home/utkarsh/Projects/PillarClassification/model/model_01_epochs_500_batchsize_1000_trainAcc_99.65_testAcc_99.60.dat'
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
ax1.plot(x,lossTrain,label='train')
ax1.plot(x,lossTest,label='test')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Loss')
ax1.set_xlim(min(x),max(x))
ax1.set_ylim(min(min(lossTrain),min(lossTest)),max(max(lossTrain),max(lossTest)))
ax1.legend()
ax2.plot(x,accuracyTrain,label='train')
ax2.plot(x,accuracyTest,label='test')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Accuracy')
ax2.set_xlim(min(x),max(x))
ax2.set_ylim(min(min(accuracyTrain),min(accuracyTest)),max(max(accuracyTrain),max(accuracyTest)))
ax2.legend()
plt.savefig(fileName,format='png')
plt.close()
