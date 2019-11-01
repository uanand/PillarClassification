import os
import sys
# import numpy
# import cv2
# from tensorflow import keras
# from tensorflow.keras import layers,optimizers
# import keras
# import matplotlib.pyplot as plt
# from keras.utils import np_utils
# from keras.models import Sequential,load_model
# from keras.layers import Dense,Dropout,Flatten,Activation
# from keras.layers import Conv2D,MaxPooling2D
# from keras import backend as K
# from keras.optimizers import SGD

sys.path.append(os.path.abspath('../lib'))
import loadData
import buildModel

############################################################
# LOAD THE LABELLED DATASET AND SPLIT INTO TRAINING AND TEST
############################################################
xTrain,yTrain,yTrainInd,xTest,yTest,yTestInd = loadData.loadPillarData(fileName='../dataset/labelledDataset.dat',numClasses=2)
############################################################

# # ############################################################
# # # TRAINING USING MODEL 1
# # ############################################################
# batch_size = 1000
# epochs = 50
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(row,col,1)))
# model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', input_shape=(row,col,1)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer=SGD(0.01), metrics=['accuracy'])
# print(model.summary())

# history = model.fit(x_train, y_train,
          # batch_size=batch_size,
          # epochs=epochs,
          # verbose=1,
          # validation_data=(x_test,y_test))

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# modelName = '../model/model1_batchsize_'+str(batch_size)+'_epochs_'+str(epochs)+'.h5'
# plotName = '../model/model1_batchsize_'+str(batch_size)+'_epochs_'+str(epochs)+'.png'
# model.save(modelName)
# ############################################################


# ############################################################
# # TRAINING USING MODEL 2
# ############################################################
# batch_size = 1000
# epochs = 200
# model = Sequential()
# model.add(Conv2D(32, (3,3), padding='same', input_shape=(row,col,1)))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, (3,3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

# opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# print(model.summary())

# history = model.fit(x_train, y_train,
          # batch_size=batch_size,
          # epochs=epochs,
          # validation_data=(x_test, y_test),
          # shuffle=True)
# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])
# modelName = '../model/model2_batchsize_'+str(batch_size)+'_epochs_'+str(epochs)+'.h5'
# plotName = '../model/model2_batchsize_'+str(batch_size)+'_epochs_'+str(epochs)+'.png'
# model.save(modelName)
# ############################################################


# ############################################################
# # TRAINING USING MODEL 3
# ############################################################
# batch_size = 1000
# epochs = 500
# model = Sequential()
# model.add(Conv2D(32, (3,3), padding='same', input_shape=(row,col,1)))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, (3,3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy', optimizer=SGD(0.1), metrics=['accuracy'])
# print(model.summary())

# history = model.fit(x_train, y_train,
          # batch_size=batch_size,
          # epochs=epochs,
          # validation_data=(x_test, y_test),
          # shuffle=True)
# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])
# model.save('../model/model3_batchsize_'+str(batch_size)+'_epochs_'+str(epochs)+'.h5')
# ############################################################


# ############################################################
# # TRAINING USING MODEL 4
# ############################################################
# batch_size = 1000
# epochs = 500
# model = Sequential()
# model.add(Conv2D(32, (3,3), padding='same', input_shape=(row,col,1)))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, (3,3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, (3,3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy', optimizer=SGD(1), metrics=['accuracy'])
# print(model.summary())

# history = model.fit(x_train, y_train,
          # batch_size=batch_size,
          # epochs=epochs,
          # validation_data=(x_test, y_test),
          # shuffle=True)
# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])
# modelName = '../model/model4_batchsize_'+str(batch_size)+'_epochs_'+str(epochs)+'.h5'
# plotName = '../model/model4_batchsize_'+str(batch_size)+'_epochs_'+str(epochs)+'.png'
# model.save(modelName)
# ############################################################


############################################################
# TRAINING USING MODEL 5
############################################################
# epochs=500
# batchSize=500
# # optimizer = optimizers.RMSprop(learning_rate=0.0001,rho=0.9,momentum=0.0,epsilon=1e-07,centered=False)
# optimizer = optimizers.SGD(learning_rate=0.0001,momentum=0.99,nesterov=False)

# model = keras.Sequential()

# model.add(layers.Input(shape=(32,32,1)))
# model.add(layers.Conv2D(filters=32,kernel_size=(3,3)))
# model.add(layers.BatchNormalization())
# model.add(layers.Activation('relu'))
# model.add(layers.MaxPooling2D(pool_size=(2,2)))
# model.add(layers.Dropout(0.25))

# model.add(layers.Conv2D(filters=64,kernel_size=(3,3)))
# model.add(layers.BatchNormalization())
# model.add(layers.Activation('relu'))
# model.add(layers.MaxPooling2D(pool_size=(2,2)))
# model.add(layers.Dropout(0.25))

# model.add(layers.Conv2D(filters=128,kernel_size=(3,3)))
# model.add(layers.BatchNormalization())
# model.add(layers.Activation('relu'))
# model.add(layers.MaxPooling2D(pool_size=(2,2)))
# model.add(layers.Dropout(0.25))

# model.add(layers.Flatten())
# model.add(layers.Dense(512))
# model.add(layers.Activation('relu'))
# model.add(layers.Dropout(0.25))

# model.add(layers.Dense(2))
# model.add(layers.Activation('softmax'))

# model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
# callbacks_list = [keras.callbacks.ModelCheckpoint('../model/model5_batchsize_intermediate_best.h5',monitor='val_accuracy',verbose=0,save_best_only=True,mode='max')]
# print(model.summary())

# history = model.fit(xTrain,yTrainInd,epochs=epochs,batch_size=batchSize,validation_data=(xTest,yTestInd),callbacks=callbacks_list)

# bestModel = keras.models.load_model('../model/model5_batchsize_intermediate_best.h5')
# scoresTrain = bestModel.evaluate(xTrain,yTrainInd,verbose=0)
# scoresTest = bestModel.evaluate(xTest,yTestInd,verbose=0)
# print ("BEST MODEL STATISTICS\nTrain loss: %f, Train accuracy: %f\nTest loss: %f, Test accuracy: %f" %(scoresTrain[0],scoresTrain[1],scoresTest[0],scoresTest[1]))

#######
# plotFileName = '../model/model5_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.png' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)
# modelFileName = '../model/model5_epochs_%d_batchsize_%d_trainAcc_%.2f_testAcc_%.2f.h5' %(epochs,batchSize,history.history['accuracy'][-1]*100,history.history['val_accuracy'][-1]*100)

# lossTrain,accuracyTrain = history.history['loss'],history.history['accuracy']
# lossTest,accuracyTest = history.history['val_loss'],history.history['val_accuracy']
# x = range(1,len(lossTrain)+1)

# fig = plt.figure(figsize=(8,4))
# ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122)

# ax1.plot(x,lossTrain,label='train')
# ax1.plot(x,lossTest,label='test')
# ax1.set_xlabel('Iterations')
# ax1.set_ylabel('Loss')
# ax1.set_xlim(min(x),max(x))
# ax1.set_ylim(min(min(lossTrain),min(lossTest)),max(max(lossTrain),max(lossTest)))
# ax1.legend()

# ax2.plot(x,accuracyTrain,label='train')
# ax2.plot(x,accuracyTest,label='test')
# ax2.set_xlabel('Iterations')
# ax2.set_ylabel('Accuracy')
# ax2.set_xlim(min(x),max(x))
# ax2.set_ylim(min(min(accuracyTrain),min(accuracyTest)),max(max(accuracyTrain),max(accuracyTest)))
# ax2.legend()

# plt.savefig(plotFileName,format='png')
# model.save(modelFileName)
# plt.close()
############################################################


############################################################
# TRAINING USING MODEL 6
############################################################
buildModel.model_06(name='model_06',xTrain=xTrain,yTrain=yTrain,yTrainInd=yTrainInd,xTest=xTest,yTest=yTest,yTestInd=yTestInd,epochs=200,batchSize=1000)
############################################################


# ############################################################
# # PLOTTING THE RESULT
# ############################################################
# history_dict = history.history
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
# acc_values = history_dict['accuracy']
# val_acc_values = history_dict['val_accuracy']
# epochs = range(1,len(loss_values)+1)

# fig = plt.figure(figsize=(12,5))
# ax1 = fig.add_subplot(1,2,1)
# ax2 = fig.add_subplot(1,2,2)
# ax1.plot(epochs,loss_values,label='Training loss')
# ax1.plot(epochs,val_loss_values,label='Validation loss')
# ax1.set_xlabel('Epochs')
# ax1.set_ylabel('Loss')
# ax1.legend()
# ax2.plot(epochs,acc_values,label='Training accuracy')
# ax2.plot(epochs,val_acc_values,label='Validation accuracy')
# ax2.set_xlabel('Epochs')
# ax2.set_ylabel('Accuracy')
# ax2.legend()
# plt.savefig(plotName,format='png')
# plt.close()
############################################################

# ############################################################
# # RUN THE MODEL ON TRAIN AND TEST DATASETS
# ############################################################
# # model = load_model('../model/model2_batchsize_1000_epochs_200.h5')
# # counter = 1
# # for i in range(num_trainData/4):
    # # gImg = x_train[i,:,:,:]
    # # gImg = numpy.reshape(gImg,(1,32,32,1))
    # # gImgRaw = (numpy.reshape(gImg,(32,32))*255).astype('uint8')
    
    # # if (y_train[i,0]==1):
        # # assignedLabel = 'Collapse'
    # # else:
        # # assignedLabel = 'Not collapse'
    # # res = model.predict_classes(gImg,batch_size=1)[0]
    # # if (res==0):
        # # predictLabel = 'Collapse'
    # # elif (res==1):
        # # predictLabel = 'Not collapse'
    # # if (assignedLabel!=predictLabel):
        # # print('trainData',counter,i,assignedLabel,predictLabel)
        # # counter+=1
        # # cv2.imwrite('/home/utkarsh/Projects/PillarClassification/dataset/incorrectClassifications/train/'+str(i).zfill(6)+'.png',gImgRaw)
    
# # counter = 1
# # for i in range(num_testData/4):
    # # gImg = x_test[i,:,:,:]
    # # gImg = numpy.reshape(gImg,(1,32,32,1))
    # # gImgRaw = (numpy.reshape(gImg,(32,32))*255).astype('uint8')
    # # if (y_test[i,0]==1):
        # # assignedLabel = 'Collapse'
    # # else:
        # # assignedLabel = 'Not collapse'
    # # res = model.predict_classes(gImg,batch_size=1)[0]
    # # if (res==0):
        # # predictLabel = 'Collapse'
    # # elif (res==1):
        # # predictLabel = 'Not collapse'
    # # if (assignedLabel!=predictLabel):
        # # print('testData',counter,i,assignedLabel,predictLabel)
        # # counter+=1
        # # cv2.imwrite('/home/utkarsh/Projects/PillarClassification/dataset/incorrectClassifications/test/'+str(i).zfill(6)+'.png',gImgRaw)
# ############################################################
