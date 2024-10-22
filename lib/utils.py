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
    '''
    Select the best model from all the intermediate saved model. Best
    model is selected by checking the accuracy and loss for all the
    training and test datasets. The best model is the one which has the
    highest average accuracy and the lowest loss and is saved with the
    test and training accuracy appened in the file name. The models
    which have a lower accuracy and higher loss are deleted. Since the
    classification is performed for all the datasets the process is
    slow.
    
    Input parameters:
    modelFileList : (list) Name with full path of the models that have
        to be tested.
    xTrain : (4D array) Normalized training dataset of shape
        [N,row,col,1(3)]. Usually the intensity values of each image are
        normalized between 0 and 1.
    yTrain : (1D array) Classification label for each training image.
        The size of this array is N.
    yTrainInd : (2D array) Indicator matrix for the training dataset.
        Its shape is [N,numClasses].
    xTest : (4D array) Normalized training dataset of shape
        [N,row,col,1(3)]. Usually the intensity values of each image are
        normalized between 0 and 1.
    yTest : (1D array) Classification label for each training image.
        The size of this array is N.
    yTestInd : (2D array) Indicator matrix for the training dataset.
        Its shape is [N,numClasses].
    '''
    bestAccuracyScore,bestLossScore = 0,1e10
    nTrain,nTest = xTrain.shape[0],xTest.shape[0]
    for modelFile in tqdm(modelFileList):
        model = keras.models.load_model(modelFile)
        scoresTrain = model.evaluate(xTrain,yTrainInd,verbose=0)
        scoresTest = model.evaluate(xTest,yTestInd,verbose=0)
        scoreAccuracy = (scoresTrain[1]+scoresTest[1])/2.0
        scoreLoss = (scoresTrain[0]+scoresTest[0])/2.0
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
    '''
    Select the best model from all the intermediate saved modelbased on
    the training and test history. The best model is the one which has
    the highest average accuracy and the lowest loss and is saved with
    the test and training accuracy appened in the file name. This best
    model selection method is faster, albeit, less accurate.
    
    Input parameters:
    modelFileList : (list) Name with full path of the models that have
        to be tested.
    history : (dict) Tensorflow/Keras history dictionary that saves the
        training and test accuracy for all the intermediate models.
    '''
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
    '''
    Convert classification labelled array y to indicator matrix.
    
    Input parameters:
    yTest : (1D array) Classification labels for N labelled images.
    numClasses : (int) Number of classes in the labelled dataset. For
        the nanopillar classification, it is 2.
    
    Returns:
    yInd : (2D array) Indicator matrix for the labelled dataset.
        Its shape is [N,numClasses].
    '''
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
    '''
    Read the Keras training history dictionary and extract the training
    accuracy, test accuracy, training loss, and test loss for each epoch
    as an array.
    
    Input parameters:
    inputFile : (dict) Tensorflow/Keras history dictionary that saves
        the training and test accuracy for all the intermediate models.
    
    Returns:
    lossTrain : (1D array) Training loss for different epochs.
    accuracyTrain : (1D array) Training accuracy for different epochs.
    lossTest : (1D array) Test loss for different epochs.
    accuracyTest : (1D array) Test loss for different epochs.
    '''
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
    '''
    Generate the images with incorrect classification. Can be used to
    check the manual labels and well as to understand the features of
    images that are being incorectly classified. At the end, it also
    prints the accuracy of the model after classifiying all the images.
    
    Input parameters:
    modelPath : (str) Keras model with full path that you want to test.
    imagePath : (str) The folder where incorrectly classified images
        will be saved.
    X : (4D array) Labelled dataset that you want to test the model on.
    Y : (1D array) Labels assigned to all the datasets.
    '''
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
    '''
    Save the manually labelled nanopillars into two seprate folders - 
    collapsed and notCollapsed. 
    
    Input parameters:
    fileName : (str) Test file containing the labelled dataset.
    numClasses : (int) Number of classes in the labelled dataset. In
        this case it was 2.
    dirList : (list of str) The list of directories where the nanopillar
        image should be saved. In this case, the labelled images were
        either saved in the collapsed or notCollapsed directory.
    '''
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
    '''
    Save the TensorFlow/Keras history dictionary as a text file.
    
    Input parameters:
    fileName : (str) File name with full path of the text file.
    history : (dict) TensorFlow/Keras history dictionary that saves
        the training and test accuracy for all the intermediate models.
    '''
    f = open(fileName,'w')
    f.write(str(history.history))
    f.close()
############################################################

############################################################
# TEST THE MODEL ACCURACY FOR TRAINING AND TEST DATASETS
############################################################
def modelAccuracy(modelFile,xTrain,yTrainInd,xTest,yTestInd):
    '''
    Test the model accuracy for the entire labelled and augmented
    dataset.
    
    Input parameters:
    modelFile : (str) Name with full path of the model that has to be
        tested.
    xTrain : (4D array) Normalized training dataset of shape
        [N,row,col,1(3)]. Usually the intensity values of each image are
        normalized between 0 and 1.
    yTrain : (1D array) Classification label for each training image.
        The size of this array is N.
    yTrainInd : (2D array) Indicator matrix for the training dataset.
        Its shape is [N,numClasses].
    xTest : (4D array) Normalized training dataset of shape
        [N,row,col,1(3)]. Usually the intensity values of each image are
        normalized between 0 and 1.
    yTest : (1D array) Classification label for each training image.
        The size of this array is N.
    yTestInd : (2D array) Indicator matrix for the training dataset.
        Its shape is [N,numClasses].
    '''
    nTrain,nTest = xTrain.shape[0],xTest.shape[0]
    model = keras.models.load_model(modelFile)
    scoresTrain = model.evaluate(xTrain,yTrainInd,verbose=0)
    scoresTest = model.evaluate(xTest,yTestInd,verbose=0)
    keras.backend.clear_session()
    print (scoresTrain,scoresTest)
############################################################
