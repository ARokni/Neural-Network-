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
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from keras.regularizers import l2

daily_model = load_model('daily_model.h5')
weekly_model = load_model('weekly_model.h5')
monthly_model = load_model('monthly_model.h5')

def split(x,y,ratio):
    tmp = np.int16((1-ratio)*len(x))
    return x[0:tmp,:],x[tmp:,:],y[0:tmp],y[tmp:]
data = np.load('polution_dataSet.npy')
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
hour = np.random.randint(0,24)
day = np.random.randint(0,7)
hours_passed_from_beinning_of_week =  24*day+hour
y_data = []
daily_x_data = []
weekly_x_data = []
monthly_x_data = []
for i in range(504,len(data)-168,168):
    tmp = i + hours_passed_from_beinning_of_week
    y_data.append(data[tmp,0])
    daily_x_data.append(data[tmp-11:tmp,:])
    x = []
    for j in range(6):
        x.append(data[tmp-24*(j+1),:])
    weekly_x_data.append(x)
    x = []
    for j in range(3):
        x.append(data[tmp-168*(j+1),:])
    monthly_x_data.append(x)
y_data = np.array(y_data)
daily_pred = daily_model.predict(np.array(daily_x_data))
weekly_pred = weekly_model.predict(np.array(weekly_x_data))
monthly_pred = monthly_model.predict(np.array(monthly_x_data))
x_train = np.concatenate((daily_pred,weekly_pred,monthly_pred),axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
x_train = scaler.fit_transform(x_train)
x_train, x_test,y_train, y_test = split(x_train,y_data,0.1)

model = Sequential()
model.add(Dense(30, kernel_regularizer=l2(0.0001), input_dim=3,activation='relu'))
model.add(Dense(30, kernel_regularizer=l2(0.0001), activation='relu'))
model.add(Dense(1,activation='relu'))
model.compile(loss='mean_absolute_error', optimizer="adam")
hist = model.fit(x_train, y_train, epochs=200, batch_size=8, validation_data = (x_test,y_test), verbose=1)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Fusion Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

fig_Accuracy = plt.figure(figsize=(25, 6))
y_pred = model.predict(x_test)
plt.plot(y_pred,label = 'prediction')
plt.plot(y_test, label = 'real data')
plt.title('Fusion model prediction vs real values')
plt.legend()