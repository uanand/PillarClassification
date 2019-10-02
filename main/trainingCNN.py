import numpy
import cv2
import keras
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import SGD

############################################################
# LOAD THE LABELLED DATASET AND SPLIT INTO TRAINING AND TEST
############################################################
labelledDatasetFile = '../dataset/labelledDataset.dat'
labelledDataset = numpy.loadtxt(labelledDatasetFile,skiprows=1)
numpy.random.shuffle(labelledDataset)
[numLabelledDataset,temp] = labelledDataset.shape

x_train,y_train = labelledDataset[:int(90.0/100*numLabelledDataset),1:],labelledDataset[:int(90.0/100*numLabelledDataset),0]
x_test,y_test = labelledDataset[int(90.0/100*numLabelledDataset):,1:],labelledDataset[int(90.0/100*numLabelledDataset):,0]
x_train = numpy.reshape(x_train,(x_train.shape[0],64,64,1))
x_test = numpy.reshape(x_test,(x_test.shape[0],64,64,1))
x_train,x_test = x_train.astype('float32'),x_test.astype('float32')
x_train/=255; x_test/=255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]
############################################################

############################################################
# TRAINING USING MODEL 1
############################################################
batch_size = 50
epochs = 500
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(64,64,1)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss = 'categorical_crossentropy',
              optimizer = SGD(0.01),
              metrics = ['accuracy'])
print(model.summary())

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('../model/model1_batchsize_'+str(batch_size)+'_epochs_'+str(epochs)+'.h5')
############################################################

############################################################
# PLOTTING THE RESULT
############################################################
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(loss_values) + 1)

fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.plot(epochs,loss_values,label='Training loss')
ax1.plot(epochs,val_loss_values,label='Validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax2.plot(epochs,acc_values,label='Training accuracy')
ax2.plot(epochs,val_acc_values,label='Validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
plt.savefig('../model/model1_batchsize_'+str(batch_size)+'_epochs_'+str(epochs[-1])+'.png',format='png')
plt.close()
