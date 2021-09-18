

import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from keras import regularizers
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(18, 16))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()



train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1)
test_datagen = ImageDataGenerator(rescale=1/255.)

test_generator = test_datagen.flow_from_directory(directory='gtsrb-german-traffic-sign/',
                              classes=['Test'],
                              class_mode=None,
                              shuffle=False,
                              target_size=(30, 30))
data = pd.read_csv("gtsrb-german-traffic-sign/Test.csv")
data = data.values
y_true = data[:,6]
y_true = list(y_true)
train_generator = train_datagen.flow_from_directory(
    directory='gtsrb-german-traffic-sign/Train',
    target_size=(30, 30),
    color_mode="rgb",
    batch_size=128,
    class_mode="categorical",
    subset='training',
    shuffle=True,
    seed=42)
validation_generator = train_datagen.flow_from_directory(
        'gtsrb-german-traffic-sign/Train',
        target_size=(30, 30),
        color_mode="rgb",
        batch_size=128,
        class_mode="categorical",
        shuffle=True,
        subset='validation',
        seed=42)


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(30, 30, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.1))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding = 'same'))


model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2),padding = 'same'))


model.add(Conv2D(128, (3, 3),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2),padding = 'same'))


model.add(Flatten())  
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))


model.add(Dense(43))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

hist = model.fit_generator(
        train_generator,
        steps_per_epoch=35289 // 128,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=3920 // 128)

preds = model.predict_generator(test_generator)
predicted_class_indices = preds.argmax(axis=-1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
predictions = list(map(int, predictions))
test_acc = sum((data[:,6] == predictions))/len(predictions)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.figure()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()




cm = confusion_matrix(y_true, predictions)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plot_confusion_matrix(cm, 
                      normalize    = True,
                      target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'
                                      ,'12','13','14','15','16','17','18','19','20','21',
                                      '22','23','24','25','26','27','28','29','30','31',
                                      '32','33','34','35','36','37','38','39','40','41','42'],
                      title        = "Confusion Matrix")
                      
print("test accuracy:")
print(test_acc)