import os
import numpy
import cv2
from tensorflow import keras
from tqdm import tqdm
import transform

############################################################
# SELECT THE BEST MODEL BASED ON ACCURACY AND LOSS VALUES
############################################################
def selectBestModel(modelFileList,xTrain,yTrainInd,xTest,yTestInd):
    bestAccuracyScore,bestLossScore = 0,1e10
    nTrain,nTest = xTrain.shape[0],xTest.shape[0]
    for modelFile in tqdm(modelFileList):
        model = keras.models.load_model(modelFile)
        scoresTrain = model.evaluate(xTrain,yTrainInd,verbose=0)
        scoresTest = model.evaluate(xTest,yTestInd,verbose=0)
        scoreAccuracy = (scoresTrain[1]+scoresTest[1])/2.0
        scoreLoss = (scoresTrain[0]+scoresTest[0])/2.0
        # scoreAccuracy = (scoresTrain[1]*nTrain+scoresTest[1]*nTest)/(nTrain+nTest)
        # scoreLoss = (scoresTrain[0]*nTrain+scoresTest[0]*nTest)/(nTrain+nTest)
        diffAccuracy,diffLoss = numpy.abs(scoresTrain[1]-scoresTest[1]),numpy.abs(scoresTrain[0]-scoresTest[0])
        if (scoreAccuracy-diffAccuracy>bestAccuracyScore):
            bestAccuracyScore = scoreAccuracy-diffAccuracy
            bestAccuracyModelFile = modelFile
        if (scoreLoss+diffLoss<bestLossScore):
            bestLossScore = scoreLoss+diffLoss
            bestLossModelFile = modelFile
        keras.backend.clear_session()
        
    for modelFile in modelFileList:
        if not(modelFile==bestAccuracyModelFile or modelFile==bestLossModelFile):
            os.remove(modelFile)
            
    modelAccuracy = keras.models.load_model(bestAccuracyModelFile)
    scoresTrain = modelAccuracy.evaluate(xTrain,yTrainInd,verbose=0)
    scoresTest = modelAccuracy.evaluate(xTest,yTestInd,verbose=0)
    print ("BEST ACCURACY MODEL STATISTICS\nTrain loss: %f, Train accuracy: %f\nTest loss: %f, Test accuracy: %f" %(scoresTrain[0],scoresTrain[1],scoresTest[0],scoresTest[1]))
    os.rename(bestAccuracyModelFile,bestAccuracyModelFile.replace('.h5','_accuracy_trainAcc_%.2f_testAcc_%.2f.h5' %(scoresTrain[1]*100,scoresTest[1]*100)))
    
    try:
        modelLoss = keras.models.load_model(bestLossModelFile)
        scoresTrain = modelLoss.evaluate(xTrain,yTrainInd,verbose=0)
        scoresTest = modelLoss.evaluate(xTest,yTestInd,verbose=0)
        print ("BEST LOSS MODEL STATISTICS\nTrain loss: %f, Train accuracy: %f\nTest loss: %f, Test accuracy: %f" %(scoresTrain[0],scoresTrain[1],scoresTest[0],scoresTest[1]))
        os.rename(bestLossModelFile,bestLossModelFile.replace('.h5','_loss_trainAcc_%.2f_testAcc_%.2f.h5' %(scoresTrain[1]*100,scoresTest[1]*100)))
    except:
        pass
############################################################

############################################################
# SELECT THE BEST MODEL BASED ON ACCURACY AND LOSS VALUES USING
# THE HISTORY DICTIONARY
############################################################
def selectBestModelHistory(modelFileList,history):
    bestAccuracyScore,bestLossScore = 0,1e10
    lossTrainList,accuracyTrainList,lossTestList,accuracyTestList = parseHistoryDict(history)
    for modelFile,lossTrain,accuracyTrain,lossTest,accuracyTest in zip(modelFileList,lossTrainList,accuracyTrainList,lossTestList,accuracyTestList):
        accuracyScore = (accuracyTrain+accuracyTest)/2.0
        lossScore = (lossTrain+lossTest)/2.0
        accuracyDiff,lossDiff = numpy.abs(accuracyTrain-accuracyTest),numpy.abs(lossTrain-lossTest)
        if (accuracyScore-accuracyDiff>bestAccuracyScore):
            bestAccuracyScore = accuracyScore-accuracyDiff
            bestAccuracyModelFile = modelFile
            bestTrainLoss_acc,bestTrainAccuracy_acc,bestTestLoss_acc,bestTestAccuracy_acc = lossTrain,accuracyTrain,lossTest,accuracyTest
        if (lossScore+lossDiff<bestLossScore):
            bestLossScore = lossScore+lossDiff
            bestLossModelFile = modelFile
            bestTrainLoss_loss,bestTrainAccuracy_loss,bestTestLoss_loss,bestTestAccuracy_loss = lossTrain,accuracyTrain,lossTest,accuracyTest
            
    for modelFile in modelFileList:
        if not(modelFile==bestAccuracyModelFile or modelFile==bestLossModelFile):
            os.remove(modelFile)
            
    print ("BEST ACCURACY MODEL STATISTICS\nTrain loss: %f, Train accuracy: %f\nTest loss: %f, Test accuracy: %f" %(bestTrainLoss_acc,bestTrainAccuracy_acc,bestTestLoss_acc,bestTestAccuracy_acc))
    os.rename(bestAccuracyModelFile,bestAccuracyModelFile.replace('.h5','_accuracy_trainAcc_%.2f_testAcc_%.2f.h5' %(bestTrainAccuracy_acc*100,bestTestAccuracy_acc*100)))
    
    try:
        print ("BEST LOSS MODEL STATISTICS\nTrain loss: %f, Train accuracy: %f\nTest loss: %f, Test accuracy: %f" %(bestTrainLoss_loss,bestTrainAccuracy_loss,bestTestLoss_loss,bestTestAccuracy_loss))
        os.rename(bestLossModelFile,bestLossModelFile.replace('.h5','_loss_trainAcc_%.2f_testAcc_%.2f.h5' %(bestTrainAccuracy_loss*100,bestTestAccuracy_loss*100)))
    except:
        pass
############################################################

############################################################
# CONVERT Y TO AN INDICATOR MATRIX
############################################################
def y2indicator(y,numClasses):
    N = y.size
    yInd = numpy.zeros([N,numClasses],dtype='float32')
    for i in range(N):
        yInd[i,y[i]] = 1
    return yInd
############################################################

############################################################
# READ THE HISTORY DICTIONARY SAVED AFTER MODEL RUN
############################################################
def parseHistoryDict(inputFile):
    f = open(inputFile,'r')
    text = f.read()
    if ('accuracy' in text):
        text = text.split("{'loss': [")[1]
        loss,text = text.split("], 'accuracy': [")[0],text.split("], 'accuracy': [")[1]
        accuracy,text = text.split("], 'val_loss': [")[0],text.split("], 'val_loss': [")[1]
        val_loss,text = text.split("], 'val_accuracy': [")[0],text.split("], 'val_accuracy': [")[1]
        val_accuracy,text = text.split("]}")[0],text.split("]}")[1]
    else:
        text = text.split("{'loss': [")[1]
        loss,text = text.split("], 'acc': [")[0],text.split("], 'acc': [")[1]
        accuracy,text = text.split("], 'val_loss': [")[0],text.split("], 'val_loss': [")[1]
        val_loss,text = text.split("], 'val_acc': [")[0],text.split("], 'val_acc': [")[1]
        val_accuracy,text = text.split("]}")[0],text.split("]}")[1]
    
    lossTrain = loss.split(", ")
    accuracyTrain = accuracy.split(", ")
    lossTest = val_loss.split(", ")
    accuracyTest = val_accuracy.split(", ")
    epochs = range(1,len(loss)+1)
    
    lossTrain = numpy.asarray(lossTrain,dtype='double')
    accuracyTrain = numpy.asarray(accuracyTrain,dtype='double')
    lossTest = numpy.asarray(lossTest,dtype='double')
    accuracyTest = numpy.asarray(accuracyTest,dtype='double')
    return lossTrain,accuracyTrain,lossTest,accuracyTest
############################################################

############################################################
# GENERATE IMAGES FOR FALSE DETECTION
############################################################
def falseClassificationImage(modelPath,imagePath,X,Y):
    [N,row,col,channel] = X.shape
    counter = 0
    model = keras.models.load_model(modelPath)
    for i,(x,y) in enumerate(zip(X,Y)):
        gImg = numpy.reshape(x,(1,row,col,channel))
        gImgRaw = (numpy.reshape(gImg,(row,col,channel))*255).astype('uint8')
        if (y==0):
            assignedLabel = 'Collapse'
        else:
            assignedLabel = 'Notcollapse'
        try:
            res = model.predict_classes(gImg,batch_size=1)[0]
        except:
            res = numpy.argmax(model.predict(gImg,batch_size=1))
        keras.backend.clear_session()
        if (res==0):
            predictLabel = 'Collapse'
        elif (res==1):
            predictLabel = 'Notcollapse'
        if (assignedLabel!=predictLabel):
            print(i+1,assignedLabel,predictLabel)
            cv2.imwrite(imagePath+'/'+str(i+1).zfill(6)+'.png',gImgRaw)
            counter+=1
    print ("Percentage of incorrect classification = %.2f" %(100.0*counter/N))
############################################################

############################################################
# GENERATE IMAGES FOR LABELLED DATASET
############################################################
def imagesForLabelDataset(fileName,numClasses,dirList):
    labelledDataset = numpy.loadtxt(fileName,skiprows=1)
    [numLabelledDataset,temp] = labelledDataset.shape
    x,y = labelledDataset[:,1:].astype('uint8'),labelledDataset[:,0].astype('uint8')
    x = transform.resizeDataset(x,(32,32))
    for i in tqdm(range(numLabelledDataset)):
        gImg = numpy.reshape(x[i,:],(32,32))
        cv2.imwrite(dirList[y[i]]+'/'+str(i+1).zfill(6)+'.png',gImg)
############################################################

############################################################
# SAVE THE KERAS TRAINING DICTIONARY AS A TEXT FILE
############################################################
def saveHistory(fileName,history):
    f = open(fileName,'w')
    f.write(str(history.history))
    f.close()
############################################################

############################################################
# TEST THE MODEL ACCURACY FOR TRAINING AND TEST DATASETS
############################################################
def modelAccuracy(modelFile,xTrain,yTrainInd,xTest,yTestInd):
    nTrain,nTest = xTrain.shape[0],xTest.shape[0]
    model = keras.models.load_model(modelFile)
    scoresTrain = model.evaluate(xTrain,yTrainInd,verbose=0)
    scoresTest = model.evaluate(xTest,yTestInd,verbose=0)
    keras.backend.clear_session()
    print (scoresTrain,scoresTest)
############################################################