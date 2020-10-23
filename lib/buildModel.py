from tensorflow import keras
from tensorflow.keras import layers,optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from time import time

import plot
import utils
import transform

############################################################
# TRAINING DATA USING MODEL 1
# DENSE (512), RELU, DROPOUT (0.50)
# DENSE (512), RELU, DROPOUT (0.50)
# DENSE (256), RELU, DROPOUT (0.50)
# DENSE (2), SOFTMAX
############################################################
def model_01(name,xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd,epochs,batchSize):
    '''
    Referred to as the DNN model in the manuscript.
    The structure of the model is:
    
    DENSE (512), RELU, DROPOUT (0.50)
    DENSE (512), RELU, DROPOUT (0.50)
    DENSE (256), RELU, DROPOUT (0.50)
    DENSE (2), SOFTMAX
    
    The model parameters are saved after every epoch. After the last
    epoch, the accuracy of all the intermediate models is tested and the
    most accuracte model is retained. All other intermediate models are
    removed.
    
    Input parameters:
    name : (str) Name that the user assigns to the model. All the
        intermediate and best models are saved with this prefix.
    xTrain : training dataset stack represented as a 4D numpy array.
        xTrain.shape yields [N, row, col, channel] where N is the number
        of images in the training dataset, row and col correspond to the
        size of the image, and channel is 1 for this model.
    yTrain : 1D array of labels. 0 is collapsed, 1 is upright.
    yTrainInd : 2D indicator array for the training dataset.
    xTest : 4D test dataset.
    yTest : 1D array of labels for the test dataset.
    yTestInd: 2D indicator array for the test dataset.
    epochs : (int) number of iterations for which the training needs to
        be done.
    batchSize : (int) number of images to use in every batch.
    '''
    optimizer = optimizers.SGD(learning_rate=0.0001,momentum=0.99,nesterov=False)
    [N,row,col,channel] = xTrain.shape
    
    model = keras.Sequential()
    model.add(layers.Input(shape=(row,col,channel)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.50))
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.50))
    model.add(layers.Dense(256))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.50))
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    filepath='../model/'+name+'_intermediate_{epoch:03d}.h5'
    callbacks_list = [keras.callbacks.ModelCheckpoint('../model/'+name+'_intermediate_{epoch:03d}.h5',monitor='val_accuracy',verbose=0,save_best_only=False,mode='auto',save_freq='epoch')]
    print(model.summary())
    
    tic = time()
    history = model.fit(xTrain,yTrainInd,epochs=epochs,batch_size=batchSize,validation_data=(xTest,yTestInd),callbacks=callbacks_list)
    toc = time(); print ('Time required %f' %(toc-tic))
    plotFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.png' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    modelFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.h5' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    historyFileName = plotFileName.replace('.png','.dat')
    utils.saveHistory(historyFileName,history)
    model.save(modelFileName)
    plot.plotMetrics(plotFileName,history)
    keras.backend.clear_session()
    
    modelFileList = []
    for epoch in range(1,epochs+1):
        modelFileList.append('../model/'+name+'_intermediate_'+str(epoch).zfill(3)+'.h5')
    utils.selectBestModelHistory(modelFileList,historyFileName)
############################################################


############################################################
# TRAINING DATA USING MODEL 2
# CONV (32,5,5,SAME), RELU, CONV (32,5,5), RELU, MAXPOOL (2,2), DROPOUT (0.50)
# DENSE (256), RELU, DROPOUT (0.50)
# DENSE (128), RELU, DROPOUT (0.50)
# DENSE (2), SOFTMAX
############################################################
def model_02(name,xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd,epochs,batchSize):
    '''
    Referred to as the CNN model in the manuscript.
    The structure of the model is:
    
    CONV (32,5,5,SAME), RELU, CONV (32,5,5), RELU, MAXPOOL (2,2), DROPOUT (0.50)
    DENSE (256), RELU, DROPOUT (0.50)
    DENSE (128), RELU, DROPOUT (0.50)
    DENSE (2), SOFTMAX
    
    The model parameters are saved after every epoch. After the last
    epoch, the accuracy of all the intermediate models is tested and the
    most accuracte model is retained. All other intermediate models are
    removed.
    
    Input parameters:
    name : (str) Name that the user assigns to the model. All the
        intermediate and best models are saved with this prefix.
    xTrain : training dataset stack represented as a 4D numpy array.
        xTrain.shape yields [N, row, col, channel] where N is the number
        of images in the training dataset, row and col correspond to the
        size of the image, and channel is 1 for this model.
    yTrain : 1D array of labels. 0 is collapsed, 1 is upright.
    yTrainInd : 2D indicator array for the training dataset.
    xTest : 4D test dataset.
    yTest : 1D array of labels for the test dataset.
    yTestInd: 2D indicator array for the test dataset.
    epochs : (int) number of iterations for which the training needs to
        be done.
    batchSize : (int) number of images to use in every batch.
    '''
    optimizer = optimizers.SGD(learning_rate=0.01,momentum=0.99,nesterov=False)
    [N,row,col,channel] = xTrain.shape
    
    model = keras.Sequential()
    model.add(layers.Input(shape=(row,col,channel)))
    model.add(layers.Conv2D(32,kernel_size=(5,5),padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32,kernel_size=(5,5)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.50))
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.50))
    model.add(layers.Dense(128))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.50))
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    filepath='../model/'+name+'_intermediate_{epoch:03d}.h5'
    callbacks_list = [keras.callbacks.ModelCheckpoint('../model/'+name+'_intermediate_{epoch:03d}.h5',monitor='val_accuracy',verbose=0,save_best_only=False,mode='auto',save_freq='epoch')]
    print(model.summary())
    
    tic = time()
    history = model.fit(xTrain,yTrainInd,epochs=epochs,batch_size=batchSize,validation_data=(xTest,yTestInd),callbacks=callbacks_list)
    toc = time(); print ('Time required %f' %(toc-tic))
    plotFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.png' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    modelFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.h5' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    historyFileName = plotFileName.replace('.png','.dat')
    utils.saveHistory(historyFileName,history)
    model.save(modelFileName)
    plot.plotMetrics(plotFileName,history)
    keras.backend.clear_session()
    
    modelFileList = []
    for epoch in range(1,epochs+1):
        modelFileList.append('../model/'+name+'_intermediate_'+str(epoch).zfill(3)+'.h5')
    utils.selectBestModelHistory(modelFileList,historyFileName)
############################################################


############################################################
# TRAINING DATA USING MODEL 3 (SAME AS MODEL 6)
# CONV (32,5,5,SAME), RELU, CONV (32,5,5), RELU, MAXPOOL (2,2), DROPOUT (0.50)
# CONV (64,5,5,SAME), RELU, CONV (64,5,5), RELU, MAXPOOL (2,2), DROPOUT (0.50)
# DENSE (256), RELU, DROPOUT (0.50)
# DENSE (128), RELU, DROPOUT (0.50)
# DENSE (2), SOFTMAX
############################################################
def model_03(name,xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd,epochs,batchSize):
    '''
    Similar to the CNN model with 2 additional convolution layers. The
    accuracy of this model was similar to CNN model.
    The structure of the model is:
    
    CONV (32,5,5,SAME), RELU, CONV (32,5,5), RELU, MAXPOOL (2,2), DROPOUT (0.50)
    CONV (64,5,5,SAME), RELU, CONV (64,5,5), RELU, MAXPOOL (2,2), DROPOUT (0.50)
    DENSE (256), RELU, DROPOUT (0.50)
    DENSE (128), RELU, DROPOUT (0.50)
    DENSE (2), SOFTMAX
    
    The model parameters are saved after every epoch. After the last
    epoch, the accuracy of all the intermediate models is tested and the
    most accuracte model is retained. All other intermediate models are
    removed.
    
    Input parameters:
    name : (str) Name that the user assigns to the model. All the
        intermediate and best models are saved with this prefix.
    xTrain : training dataset stack represented as a 4D numpy array.
        xTrain.shape yields [N, row, col, channel] where N is the number
        of images in the training dataset, row and col correspond to the
        size of the image, and channel is 1 for this model.
    yTrain : 1D array of labels. 0 is collapsed, 1 is upright.
    yTrainInd : 2D indicator array for the training dataset.
    xTest : 4D test dataset.
    yTest : 1D array of labels for the test dataset.
    yTestInd: 2D indicator array for the test dataset.
    epochs : (int) number of iterations for which the training needs to
        be done.
    batchSize : (int) number of images to use in every batch.
    '''
    optimizer = optimizers.SGD(learning_rate=0.001,momentum=0.99,nesterov=False)
    [N,row,col,channel] = xTrain.shape
    
    model = keras.Sequential()
    model.add(layers.Input(shape=(row,col,channel)))
    model.add(layers.Conv2D(32, kernel_size=(5,5), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, kernel_size=(5,5)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.50))
    model.add(layers.Conv2D(64, kernel_size=(5,5), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, kernel_size=(5,5)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.50))
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.50))
    model.add(layers.Dense(128))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.50))
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    filepath='../model/'+name+'_intermediate_{epoch:03d}.h5'
    callbacks_list = [keras.callbacks.ModelCheckpoint('../model/'+name+'_intermediate_{epoch:03d}.h5',monitor='val_accuracy',verbose=0,save_best_only=False,mode='auto',save_freq='epoch')]
    print(model.summary())
    
    history = model.fit(xTrain,yTrainInd,epochs=epochs,batch_size=batchSize,validation_data=(xTest,yTestInd),callbacks=callbacks_list)
    plotFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.png' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    modelFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.h5' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    historyFileName = plotFileName.replace('.png','.dat')
    utils.saveHistory(historyFileName,history)
    model.save(modelFileName)
    plot.plotMetrics(plotFileName,history)
    keras.backend.clear_session()
    
    modelFileList = []
    for epoch in range(1,epochs+1):
        modelFileList.append('../model/'+name+'_intermediate_'+str(epoch).zfill(3)+'.h5')
    utils.selectBestModelHistory(modelFileList,historyFileName)
############################################################


############################################################
# TRAINING DATA USING MODEL 4
# CONV (32,3,3), BATCHNORMALIZATION, RELU, MAXPOOL (2,2), DROPOUT (0.50)
# CONV (64,3,3), BATCHNORMALIZATION, RELU, MAXPOOL (2,2), DROPOUT (0.50)
# CONV (128,3,3), BATCHNORMALIZATION, RELU, MAXPOOL (2,2), DROPOUT (0.50)
# DENSE (512), RELU, DROPOUT (0.50)
# DENSE (2), SOFTMAX
############################################################
def model_04(name,xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd,epochs,batchSize):
    '''
    Similar to the CNN model with 3 convolution layers of different
    size. The accuracy of this model was similar to CNN model.
    The structure of the model is:
    
    CONV (32,3,3), BATCHNORMALIZATION, RELU, MAXPOOL (2,2), DROPOUT (0.50)
    CONV (64,3,3), BATCHNORMALIZATION, RELU, MAXPOOL (2,2), DROPOUT (0.50)
    CONV (128,3,3), BATCHNORMALIZATION, RELU, MAXPOOL (2,2), DROPOUT (0.50)
    DENSE (512), RELU, DROPOUT (0.50)
    DENSE (2), SOFTMAX
    
    The model parameters are saved after every epoch. After the last
    epoch, the accuracy of all the intermediate models is tested and the
    most accuracte model is retained. All other intermediate models are
    removed.
    
    Input parameters:
    name : (str) Name that the user assigns to the model. All the
        intermediate and best models are saved with this prefix.
    xTrain : training dataset stack represented as a 4D numpy array.
        xTrain.shape yields [N, row, col, channel] where N is the number
        of images in the training dataset, row and col correspond to the
        size of the image, and channel is 1 for this model.
    yTrain : 1D array of labels. 0 is collapsed, 1 is upright.
    yTrainInd : 2D indicator array for the training dataset.
    xTest : 4D test dataset.
    yTest : 1D array of labels for the test dataset.
    yTestInd: 2D indicator array for the test dataset.
    epochs : (int) number of iterations for which the training needs to
        be done.
    batchSize : (int) number of images to use in every batch.
    '''
    optimizer = optimizers.SGD(learning_rate=0.01,momentum=0.99,nesterov=False)
    [N,row,col,channel] = xTrain.shape
    
    model = keras.Sequential()
    model.add(layers.Input(shape=(row,col,channel)))
    model.add(layers.Conv2D(filters=32,kernel_size=(3,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.50))
    model.add(layers.Conv2D(filters=64,kernel_size=(3,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.50))
    model.add(layers.Conv2D(filters=128,kernel_size=(3,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.50))
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.50))
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    filepath='../model/'+name+'_intermediate_{epoch:03d}.h5'
    callbacks_list = [keras.callbacks.ModelCheckpoint('../model/'+name+'_intermediate_{epoch:03d}.h5',monitor='val_accuracy',verbose=0,save_best_only=False,mode='auto',save_freq='epoch')]
    print(model.summary())
    
    history = model.fit(xTrain,yTrainInd,epochs=epochs,batch_size=batchSize,validation_data=(xTest,yTestInd),callbacks=callbacks_list)
    plotFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.png' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    modelFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.h5' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    historyFileName = plotFileName.replace('.png','.dat')
    utils.saveHistory(historyFileName,history)
    model.save(modelFileName)
    plot.plotMetrics(plotFileName,history)
    keras.backend.clear_session()
    
    modelFileList = []
    for epoch in range(1,epochs+1):
        modelFileList.append('../model/'+name+'_intermediate_'+str(epoch).zfill(3)+'.h5')
    utils.selectBestModelHistory(modelFileList,historyFileName)
############################################################


############################################################
# TRAINING DATA USING MODEL 5
# CONV (32,3,3), RELU, SAME, CONV (32,3,3), RELU, MAXPOOL (2,2), DROPOUT (0.50)
# CONV (64,3,3), RELU, SAME, CONV (64,3,3), RELU, MAXPOOL (2,2), DROPOUT (0.50)
# DENSE (512), RELU, DROPOUT (0.50)
# DENSE (2), SOFTMAX
############################################################
def model_05(name,xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd,epochs,batchSize):
    '''
    4 convoluton layers with relu activation and dropout after every 2nd
    layer. No batcnormalization performed. This model was had a
    relatively low prediction accuracy. The structure of the model is:
    
    CONV (32,3,3), RELU, SAME, CONV (32,3,3), RELU, MAXPOOL (2,2), DROPOUT (0.50)
    CONV (64,3,3), RELU, SAME, CONV (64,3,3), RELU, MAXPOOL (2,2), DROPOUT (0.50)
    DENSE (512), RELU, DROPOUT (0.50)
    DENSE (2), SOFTMAX
    
    The model parameters are saved after every epoch. After the last
    epoch, the accuracy of all the intermediate models is tested and the
    most accuracte model is retained. All other intermediate models are
    removed.
    
    Input parameters:
    name : (str) Name that the user assigns to the model. All the
        intermediate and best models are saved with this prefix.
    xTrain : training dataset stack represented as a 4D numpy array.
        xTrain.shape yields [N, row, col, channel] where N is the number
        of images in the training dataset, row and col correspond to the
        size of the image, and channel is 1 for this model.
    yTrain : 1D array of labels. 0 is collapsed, 1 is upright.
    yTrainInd : 2D indicator array for the training dataset.
    xTest : 4D test dataset.
    yTest : 1D array of labels for the test dataset.
    yTestInd: 2D indicator array for the test dataset.
    epochs : (int) number of iterations for which the training needs to
        be done.
    batchSize : (int) number of images to use in every batch.
    '''
    optimizer = optimizers.SGD(learning_rate=0.01,momentum=0.99,nesterov=False)
    [N,row,col,channel] = xTrain.shape
    
    model = keras.Sequential()
    model.add(layers.Input(shape=(row,col,channel)))
    model.add(layers.Conv2D(32, (3,3), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3,3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.50))
    model.add(layers.Conv2D(64, (3,3), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3,3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.50))
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    filepath='../model/'+name+'_intermediate_{epoch:03d}.h5'
    callbacks_list = [keras.callbacks.ModelCheckpoint('../model/'+name+'_intermediate_{epoch:03d}.h5',monitor='val_accuracy',verbose=0,save_best_only=False,mode='auto',save_freq='epoch')]
    print(model.summary())
    
    history = model.fit(xTrain,yTrainInd,epochs=epochs,batch_size=batchSize,validation_data=(xTest,yTestInd),callbacks=callbacks_list)
    plotFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.png' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    modelFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.h5' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    historyFileName = plotFileName.replace('.png','.dat')
    utils.saveHistory(historyFileName,history)
    model.save(modelFileName)
    plot.plotMetrics(plotFileName,history)
    keras.backend.clear_session()
    
    modelFileList = []
    for epoch in range(1,epochs+1):
        modelFileList.append('../model/'+name+'_intermediate_'+str(epoch).zfill(3)+'.h5')
    utils.selectBestModelHistory(modelFileList,historyFileName)
############################################################


############################################################
# USE VGG TRAINING WEIGHTS FOR CLASSIFICATION
# vgg16 = VGG16(input_shape=(row,col,channel),weights='imagenet',include_top=False)
    # for layer in vgg16.layers:
        # layer.trainable = False
    # x = layers.Flatten()(vgg16.output)
    # x = layers.Dense(100, activation='relu')(x)
    # x = layers.Dense(2, activation='softmax')(x)
    # model = keras.Model(inputs=vgg16.input, outputs=x)
############################################################
def trainUsingVGG16(name,xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd,epochs,batchSize):
    '''
    VGG16 model with additional dense layers of size 100 and 2 at the
    end. Imagenet weights wer used for the convolution section and
    training was performed only on the final 2 dense layers. The
    structure of the model is:
    
    x = layers.Flatten()(vgg16.output)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dense(2, activation='softmax')(x)
    model = keras.Model(inputs=vgg16.input, outputs=x)
    
    The model parameters are saved after every epoch. After the last
    epoch, the accuracy of all the intermediate models is tested and the
    most accuracte model is retained. All other intermediate models are
    removed. The training and test datasets have 3 channels (RGB) and
    the data renormalization is done by subtracting the mean intensity
    value from each image.
    
    Input parameters:
    name : (str) Name that the user assigns to the model. All the
        intermediate and best models are saved with this prefix.
    xTrain : training dataset stack represented as a 4D numpy array.
        xTrain.shape yields [N, row, col, channel] where N is the number
        of images in the training dataset, row and col correspond to the
        size of the image, and channel is 1 for this model.
    yTrain : 1D array of labels. 0 is collapsed, 1 is upright.
    yTrainInd : 2D indicator array for the training dataset.
    xTest : 4D test dataset.
    yTest : 1D array of labels for the test dataset.
    yTestInd: 2D indicator array for the test dataset.
    epochs : (int) number of iterations for which the training needs to
        be done.
    batchSize : (int) number of images to use in every batch.
    '''
    optimizer = optimizers.SGD(learning_rate=0.01,momentum=0.99,nesterov=False)
    [N,row,col,channel] = xTrain.shape
    xTrain,xTest = transform.renormalizeDataset(xTrain,xTest,VGG=True)
    
    vgg16 = VGG16(input_shape=(row,col,channel),weights='imagenet',include_top=False)
    for layer in vgg16.layers:
        layer.trainable = False
        
    x = layers.Flatten()(vgg16.output)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dense(2, activation='softmax')(x)
    model = keras.Model(inputs=vgg16.input, outputs=x)
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    filepath='../model/'+name+'_intermediate_{epoch:03d}.h5'
    callbacks_list = [keras.callbacks.ModelCheckpoint('../model/'+name+'_intermediate_{epoch:03d}.h5',monitor='val_accuracy',verbose=0,save_best_only=False,mode='auto',save_freq='epoch')]
    print(model.summary())
    
    tic = time()
    history = model.fit(xTrain,yTrainInd,epochs=epochs,batch_size=batchSize,validation_data=(xTest,yTestInd),callbacks=callbacks_list)
    toc = time(); print ('Time required %f' %(toc-tic))
    plotFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.png' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    modelFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.h5' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    historyFileName = plotFileName.replace('.png','.dat')
    utils.saveHistory(historyFileName,history)
    model.save(modelFileName)
    plot.plotMetrics(plotFileName,history)
    keras.backend.clear_session()
    
    modelFileList = []
    for epoch in range(1,epochs+1):
        modelFileList.append('../model/'+name+'_intermediate_'+str(epoch).zfill(3)+'.h5')
    utils.selectBestModelHistory(modelFileList,historyFileName)
############################################################


############################################################
# LOAD AN INTERMEDIATE MODEL AND IMPROVE USING DIFFERENT
# OPTIMIZATION PARAMETERS
############################################################
def trainIntermediateModel(modelFile,name,xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd,optimizer,epochs,batchSize,bestModelMode='history'):
    '''
    Train an existing model using different learning rate and optimizer.
    
    Input parameters:
    modelFile : (str) name of the intermediate model that needs to be
        trained again
    name : (str) Name that the user assigns to the model. All the
        intermediate and best models are saved with this prefix.
    xTrain : training dataset stack represented as a 4D numpy array.
        xTrain.shape yields [N, row, col, channel] where N is the number
        of images in the training dataset, row and col correspond to the
        size of the image, and channel is 1 for this model.
    yTrain : 1D array of labels. 0 is collapsed, 1 is upright.
    yTrainInd : 2D indicator array for the training dataset.
    xTest : 4D test dataset.
    yTest : 1D array of labels for the test dataset.
    yTestInd: 2D indicator array for the test dataset.
    epochs : (int) number of iterations for which the training needs to
        be done.
    batchSize : (int) number of images to use in every batch.
    bestModelMode : method for selecting the best model. Available
        options are 'history',and  'all'. On choosing 'history' the best
        model is selected using the training history dictionary. If
        'all' the best model is selected by testing the model accuracy
        for the entire dataset - training + test data.
    '''
    model = keras.models.load_model('../model/'+modelFile+'.h5')
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    callbacks_list = [keras.callbacks.ModelCheckpoint('../model/'+name+'_intermediate_{epoch:03d}.h5',monitor='val_accuracy',verbose=0,save_best_only=False,mode='auto',save_freq='epoch')]
    print(model.summary())
    
    history = model.fit(xTrain,yTrainInd,epochs=epochs,batch_size=batchSize,validation_data=(xTest,yTestInd),callbacks=callbacks_list)
    plotFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.png' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    modelFileName = '../model/'+name+'_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.h5' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
    historyFileName = plotFileName.replace('.png','.dat')
    utils.saveHistory(historyFileName,history)
    model.save(modelFileName)
    plot.plotMetrics(plotFileName,history)
    keras.backend.clear_session()
    
    modelFileList = []
    for epoch in range(1,epochs+1):
        modelFileList.append('../model/'+name+'_intermediate_'+str(epoch).zfill(3)+'.h5')
    if (bestModelMode=='histroy'):
        utils.selectBestModelHistory(modelFileList,historyFileName)
    elif (bestModelMode=='all'):
        utils.selectBestModel(modelFileList,xTrain,yTrainInd,xTest,yTestInd)
############################################################
