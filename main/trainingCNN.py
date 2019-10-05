import os
import sys
import numpy
import cv2
import keras
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,Flatten,Activation
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as K
from keras.optimizers import SGD

sys.path.append(os.path.abspath('../lib'))
import transform

############################################################
# LOAD THE LABELLED DATASET AND SPLIT INTO TRAINING AND TEST
############################################################
labelledDatasetFile = '../dataset/labelledDataset.dat'
labelledDataset = numpy.loadtxt(labelledDatasetFile,skiprows=1)
numpy.random.seed(0)
numpy.random.shuffle(labelledDataset)
[numLabelledDataset,temp] = labelledDataset.shape

x_train_original,y_train_original = labelledDataset[:4000,1:],labelledDataset[:4000,0]
x_test_original,y_test_original = labelledDataset[4000:,1:],labelledDataset[4000:,0]
x_train_rot90,y_train_rot90 = transform.rotateDataset(x_train_original,y_train_original,90)
x_test_rot90,y_test_rot90 = transform.rotateDataset(x_test_original,y_test_original,90)
x_train_rot180,y_train_rot180 = transform.rotateDataset(x_train_original,y_train_original,180)
x_test_rot180,y_test_rot180 = transform.rotateDataset(x_test_original,y_test_original,180)
x_train_rot270,y_train_rot270 = transform.rotateDataset(x_train_original,y_train_original,270)
x_test_rot270,y_test_rot270 = transform.rotateDataset(x_test_original,y_test_original,270)

x_train = numpy.row_stack((x_train_original,x_train_rot90,x_train_rot180,x_train_rot270))
y_train = numpy.concatenate((y_train_original,y_train_rot90,y_train_rot180,y_train_rot270))
x_test = numpy.row_stack((x_test_original,x_test_rot90,x_test_rot180,x_test_rot270))
y_test = numpy.concatenate((y_test_original,y_test_rot90,y_test_rot180,y_test_rot270))

x_train = transform.resizeDataset(x_train,(32,32))
x_test = transform.resizeDataset(x_test,(32,32))
row,col = 32,32

x_train = numpy.reshape(x_train,(x_train.shape[0],row,col,1))
x_test = numpy.reshape(x_test,(x_test.shape[0],row,col,1))
x_train,x_test = x_train.astype('float32'),x_test.astype('float32')
x_train/=255; x_test/=255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]
num_trainData,num_testData = x_train.shape[0],x_test.shape[0]
# ############################################################

# ############################################################
# # TRAINING USING MODEL 1
# ############################################################
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
# model.save('../model/model1_batchsize_'+str(batch_size)+'_epochs_'+str(epochs)+'.h5')
############################################################


############################################################
# TRAINING USING MODEL 2
############################################################
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

# # initiate RMSprop optimizer and configure some parameters
# opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

# # Let's create our model
# # model.compile(loss='categorical_crossentropy', optimizer=SGD(0.01), metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# print(model.summary())

# history = model.fit(x_train, y_train,
          # batch_size=batch_size,
          # epochs=epochs,
          # validation_data=(x_test, y_test))
# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])
# model.save('../model/model2_batchsize_'+str(batch_size)+'_epochs_'+str(epochs)+'.h5')
############################################################

############################################################
# PLOTTING THE RESULT
############################################################
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
# plt.savefig('../model/model1_batchsize_'+str(batch_size)+'_epochs_'+str(epochs[-1])+'.png',format='png')
# plt.close()
############################################################

############################################################
# RUN THE MODEL ON TRAIN AND TEST DATASETS
############################################################
model = load_model('../model/model2_batchsize_1000_epochs_200.h5')
counter = 1
for i in range(num_trainData/4):
    gImg = x_train[i,:,:,:]
    gImg = numpy.reshape(gImg,(1,32,32,1))
    gImgRaw = (numpy.reshape(gImg,(32,32))*255).astype('uint8')
    
    if (y_train[i,0]==1):
        assignedLabel = 'Collapse'
    else:
        assignedLabel = 'Not collapse'
    res = model.predict_classes(gImg,batch_size=1)[0]
    if (res==0):
        predictLabel = 'Collapse'
    elif (res==1):
        predictLabel = 'Not collapse'
    if (assignedLabel!=predictLabel):
        print('trainData',counter,i,assignedLabel,predictLabel)
        counter+=1
        cv2.imwrite('/home/utkarsh/Projects/PillarClassification/dataset/incorrectClassifications/train/'+str(i).zfill(6)+'.png',gImgRaw)
    
counter = 1
for i in range(num_testData/4):
    gImg = x_test[i,:,:,:]
    gImg = numpy.reshape(gImg,(1,32,32,1))
    gImgRaw = (numpy.reshape(gImg,(32,32))*255).astype('uint8')
    if (y_test[i,0]==1):
        assignedLabel = 'Collapse'
    else:
        assignedLabel = 'Not collapse'
    res = model.predict_classes(gImg,batch_size=1)[0]
    if (res==0):
        predictLabel = 'Collapse'
    elif (res==1):
        predictLabel = 'Not collapse'
    if (assignedLabel!=predictLabel):
        print('testData',counter,i,assignedLabel,predictLabel)
        counter+=1
        cv2.imwrite('/home/utkarsh/Projects/PillarClassification/dataset/incorrectClassifications/test/'+str(i).zfill(6)+'.png',gImgRaw)
