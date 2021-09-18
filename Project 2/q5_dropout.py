import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU,SimpleRNN
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import RMSprop, SGD , Adagrad , Adam
from keras.layers import Input, Flatten, Dropout, Activation,BatchNormalization
from keras.models import load_model
from keras.regularizers import l2
from sklearn.metrics import mean_absolute_error
def create_dataset(dataset, look_back=11):
    data_x, data_y = [], []
    for i in range(15000):
        data_x.append(dataset[i:i+look_back,:])
        data_y.append(dataset[i+look_back+1,0])
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    return data_x[0:1000,:],data_x[1000:1500,:],data_y[0:1000],data_y[1000:1500]



data = np.load('polution_dataSet.npy')
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
x_train,x_test,y_train,y_test = create_dataset(data,11)

model = Sequential()
model.add(GRU(100, input_shape=(11, 8)))
model.add(Dense(16))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer="adam")
hist = model.fit(x_train, y_train, epochs=550, batch_size=64,validation_data=(x_test, y_test), verbose=1)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss without dropout')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()



fig_Accuracy = plt.figure(figsize=(25, 6))
y_pred=model.predict(x_test)
plt.plot(y_pred, label = "predicted pollution")
plt.plot(y_test,label = "real pollution")
plt.legend()
plt.title('real and predicted pollution without dropout')
z = mean_absolute_error(y_test, y_pred)
plt.show()


model = Sequential()
model.add(GRU(100,recurrent_dropout =0.6, input_shape=(11, 8)))
model.add(Dense(16))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer="adam")
hist = model.fit(x_train, y_train, epochs=550, batch_size=64,validation_data=(x_test, y_test), verbose=1)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss with dropout')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()



fig_Accuracy = plt.figure(figsize=(25, 6))
y_pred=model.predict(x_test)
plt.plot(y_pred, label = "predicted pollution")
plt.plot(y_test,label = "real pollution")
plt.legend()
plt.title('real and predicted pollution without dropout')
z = mean_absolute_error(y_test, y_pred)
plt.show()
