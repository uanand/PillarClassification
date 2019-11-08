from tensorflow import keras
from tensorflow.keras import layers,optimizers
import plot
import utils

# optimizer = optimizers.Adadelta(learning_rate=0.001,rho=0.95,epsilon=1e-07)
# optimizer = optimizers.Adagrad(learning_rate=0.001,initial_accumulator_value=0.1,epsilon=1e-07)
# optimizer = optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07,amsgrad=False)
# optimizer = optimizers.Adamax(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07)
# optimizer = optimizers.Ftrl(learning_rate=0.001,learning_rate_power=-0.5,initial_accumulator_value=0.1,l1_regularization_strength=0.0,l2_regularization_strength=0.0)
# optimizer = optimizers.Nadam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07)
# optimizer = optimizers.RMSprop(learning_rate=0.001,rho=0.9,momentum=0.0,epsilon=1e-07,centered=False)
# optimizer = optimizers.SGD(learning_rate=0.01,momentum=0.99,nesterov=False)


############################################################
# TRAINING DATA USING MODEL 1
# DENSE (512), RELU, DROPOUT (0.25)
# DENSE (512), RELU, DROPOUT (0.25)
# DENSE (256), RELU, DROPOUT (0.25)
# DENSE (2), SOFTMAX
############################################################
def model_01(name,xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd,epochs,batchSize):
    optimizer = optimizers.RMSprop(learning_rate=0.0001,rho=0.9,momentum=0.0,epsilon=1e-07,centered=False)
    
    model = keras.Sequential()
    model.add(layers.Input(shape=(32,32,1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(256))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    filepath='../model/'+name+'_intermediate_{epoch:03d}.h5'
    callbacks_list = [keras.callbacks.ModelCheckpoint('../model/'+name+'_intermediate_{epoch:03d}.h5',monitor='val_accuracy',verbose=0,save_best_only=False,mode='auto',period=1)]
    print(model.summary())
    
    history = model.fit(xTrain,yTrainInd,epochs=epochs,batch_size=batchSize,validation_data=(xTest,yTestInd),callbacks=callbacks_list)
    plotFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.png' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    modelFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.h5' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    model.save(modelFileName)
    plot.plotMetrics(plotFileName,history)
    
    modelFileList = []
    for epoch in range(1,epochs+1):
        modelFileList.append('../model/'+name+'_intermediate_'+str(epoch).zfill(3)+'.h5')
    bestModelFile = utils.selectBestModel(modelFileList,xTrain,yTrainInd,xTest,yTestInd)
############################################################


############################################################
# TRAINING DATA USING MODEL 2
# CONV (32,5,5), RELU
# CONV (64,5,5), RELU, MAXPOOL (2,2), DROPOUT (0.25)
# DENSE (128), RELU, DROPOUT (0.50)
# DENSE (2), SOFTMAX
############################################################
def model_02(name,xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd,epochs,batchSize):
    optimizer = optimizers.SGD(learning_rate=0.01,momentum=0.99,nesterov=False)
    
    model = keras.Sequential()
    model.add(layers.Input(shape=(32,32,1)))
    model.add(layers.Conv2D(32,kernel_size=(5,5)))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64,kernel_size=(5,5)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    filepath='../model/'+name+'_intermediate_{epoch:03d}.h5'
    callbacks_list = [keras.callbacks.ModelCheckpoint('../model/'+name+'_intermediate_{epoch:03d}.h5',monitor='val_accuracy',verbose=0,save_best_only=False,mode='auto',period=1)]
    print(model.summary())
    
    history = model.fit(xTrain,yTrainInd,epochs=epochs,batch_size=batchSize,validation_data=(xTest,yTestInd),callbacks=callbacks_list)
    plotFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.png' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    modelFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.h5' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    model.save(modelFileName)
    plot.plotMetrics(plotFileName,history)
    
    modelFileList = []
    for epoch in range(1,epochs+1):
        modelFileList.append('../model/'+name+'_intermediate_'+str(epoch).zfill(3)+'.h5')
    bestModelFile = utils.selectBestModel(modelFileList,xTrain,yTrainInd,xTest,yTestInd)
############################################################


############################################################
# TRAINING DATA USING MODEL 3 (SAME AS MODEL 6)
# CONV (32,3,3,SAME), RELU 
# CONV (32,3,3), RELU
# MAXPOOL (2,2), DROPOUT (0.25)
# CONV (64,3,3,SAME), RELU
# CONV (64,3,3), RELU
# MAXPOOL (2,2), DROPOUT (0.25)
# DENSE (512), RELU, DROPOUT (0.50)
# DENSE (2), SOFTMAX
############################################################
def model_03(name,xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd,epochs,batchSize):
    optimizer = optimizers.RMSprop(learning_rate=0.001, decay=1e-6)
    
    model = keras.Sequential()
    model.add(layers.Input(shape=(32,32,1)))
    model.add(layers.Conv2D(32, kernel_size=(3,3), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, kernel_size=(3,3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, kernel_size=(3,3), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, kernel_size=(3,3)))
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
    filepath='../model/'+name+'_intermediate_{epoch:03d}.h5'
    callbacks_list = [keras.callbacks.ModelCheckpoint('../model/'+name+'_intermediate_{epoch:03d}.h5',monitor='val_accuracy',verbose=0,save_best_only=False,mode='auto',period=1)]
    print(model.summary())
    
    history = model.fit(xTrain,yTrainInd,epochs=epochs,batch_size=batchSize,validation_data=(xTest,yTestInd),callbacks=callbacks_list)
    plotFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.png' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    modelFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.h5' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    model.save(modelFileName)
    plot.plotMetrics(plotFileName,history)
    
    modelFileList = []
    for epoch in range(1,epochs+1):
        modelFileList.append('../model/'+name+'_intermediate_'+str(epoch).zfill(3)+'.h5')
    bestModelFile = utils.selectBestModel(modelFileList,xTrain,yTrainInd,xTest,yTestInd)
############################################################


############################################################
# TRAINING DATA USING MODEL 4
# CONV (32,3,3), BATCHNORMALIZATION, RELU, MAXPOOL (2,2), DROPOUT (0.25)
# CONV (64,3,3), BATCHNORMALIZATION, RELU, MAXPOOL (2,2), DROPOUT (0.25)
# CONV (128,3,3), BATCHNORMALIZATION, RELU, MAXPOOL (2,2), DROPOUT (0.25)
# DENSE (512), RELU, DROPOUT (0.25)
# DENSE (2), SOFTMAX
############################################################
def model_04(name,xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd,epochs,batchSize):
    optimizer = optimizers.SGD(learning_rate=0.0001,momentum=0.99,nesterov=False)
    
    model = keras.Sequential()
    model.add(layers.Input(shape=(32,32,1)))
    model.add(layers.Conv2D(filters=32,kernel_size=(3,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(filters=64,kernel_size=(3,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(filters=128,kernel_size=(3,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    filepath='../model/'+name+'_intermediate_{epoch:03d}.h5'
    callbacks_list = [keras.callbacks.ModelCheckpoint('../model/'+name+'_intermediate_{epoch:03d}.h5',monitor='val_accuracy',verbose=0,save_best_only=False,mode='auto',period=1)]
    print(model.summary())
    
    history = model.fit(xTrain,yTrainInd,epochs=epochs,batch_size=batchSize,validation_data=(xTest,yTestInd),callbacks=callbacks_list)
    plotFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.png' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    modelFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.h5' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    model.save(modelFileName)
    plot.plotMetrics(plotFileName,history)
    
    modelFileList = []
    for epoch in range(1,epochs+1):
        modelFileList.append('../model/'+name+'_intermediate_'+str(epoch).zfill(3)+'.h5')
    bestModelFile = utils.selectBestModel(modelFileList,xTrain,yTrainInd,xTest,yTestInd)
############################################################


############################################################
# TRAINING DATA USING MODEL 5
# CONV (32,3,3), RELU, SAME
# CONV (32,3,3), RELU
# MAXPOOL (2,2)
# CONV (64,3,3), RELU, SAME
# CONV (64,3,3), RELU
# MAXPOOL (2,2)
# DENSE (512), RELU
# DENSE (2), SOFTMAX
############################################################
def model_05(name,xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd,epochs,batchSize):
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
    filepath='../model/'+name+'_intermediate_{epoch:03d}.h5'
    callbacks_list = [keras.callbacks.ModelCheckpoint('../model/'+name+'_intermediate_{epoch:03d}.h5',monitor='val_accuracy',verbose=0,save_best_only=False,mode='auto',period=1)]
    print(model.summary())
    
    history = model.fit(xTrain,yTrainInd,epochs=epochs,batch_size=batchSize,validation_data=(xTest,yTestInd),callbacks=callbacks_list)
    plotFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.png' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    modelFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.h5' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    model.save(modelFileName)
    plot.plotMetrics(plotFileName,history)
    
    modelFileList = []
    for epoch in range(1,epochs+1):
        modelFileList.append('../model/'+name+'_intermediate_'+str(epoch).zfill(3)+'.h5')
    bestModelFile = utils.selectBestModel(modelFileList,xTrain,yTrainInd,xTest,yTestInd)
############################################################
