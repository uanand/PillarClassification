import os
import numpy
from tensorflow import keras
from tqdm import tqdm 

def selectBestModel(modelFileList,xTrain,yTrainInd,xTest,yTestInd):
    bestAccuracyScore,bestLossScore = 0,1e10
    # minDiffAccuracy,minDiffLoss = 1e10,1e10
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
            # if (diffAccuracy<=minDiffAccuracy):
                bestAccuracyScore = scoreAccuracy-diffAccuracy
                # bestAccuracyScore = scoreAccuracy
                bestAccuracyModelFile = modelFile
                # minDiffAccuracy = diffAccuracy
        if (scoreLoss+diffLoss<bestLossScore):
            # if (diffLoss<=minDiffLoss):
                bestLossScore = scoreLoss+diffLoss
                # bestLossScore = scoreLoss
                bestLossModelFile = modelFile
                # minDiffLoss = diffLoss
        # del model
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
