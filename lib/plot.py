import matplotlib.pyplot as plt

#######################################################################
# PLOT THE VALIDATION AND TEST LOSS AND ACCURACY FOR THE FITTED MODELS
#######################################################################
def plotMetrics(fileName,history):
    lossTrain,accuracyTrain = history.history['loss'],history.history['accuracy']
    lossTest,accuracyTest = history.history['val_loss'],history.history['val_accuracy']
    x = range(1,len(lossTrain)+1)
    
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
    
    f = open(fileName.replace('.png','.dat'),'w')
    f.write(str(history.history))
    f.close()
#######################################################################
