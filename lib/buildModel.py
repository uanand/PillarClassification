from tensorflow import keras
from tensorflow.keras import layers,optimizers

import plot

def model_06(name,xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd,epochs,batchSize):
    optimizer = optimizers.RMSprop(learning_rate=0.001,decay=1e-6)
    
    model = keras.Sequential()
    model.add(layers.Input(shape=(32,32,1)))
    model.add(layers.Conv2D(32, (3,3), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3,3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(64, (3,3), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3,3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    callbacks_list = [keras.callbacks.ModelCheckpoint('../model/'+name+'_intermediate_best.h5',monitor='val_accuracy',verbose=0,save_best_only=True,mode='max')]
    print(model.summary())
    
    history = model.fit(xTrain,yTrainInd,epochs=epochs,batch_size=batchSize,validation_data=(xTest,yTestInd),callbacks=callbacks_list)
    plotFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.png' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    modelFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.h5' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    model.save(modelFileName)
    plot.plotMetrics(plotFileName,history)
    
    bestModel = keras.models.load_model('../model/'+name+'_intermediate_best.h5')
    scoresTrain = bestModel.evaluate(xTrain,yTrainInd,verbose=0)
    scoresTest = bestModel.evaluate(xTest,yTestInd,verbose=0)
    print ("BEST MODEL STATISTICS\nTrain loss: %f, Train accuracy: %f\nTest loss: %f, Test accuracy: %f" %(scoresTrain[0],scoresTrain[1],scoresTest[0],scoresTest[1]))
    
