


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

def create_dataset(dataset, look_back=11):
    data_x, data_y = [], []
    for i in range(15000):
        data_x.append(dataset[i:i+look_back,:])
        data_y.append(dataset[i+look_back,0])
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    return data_x[0:12000,:],data_x[12000:15000,:],data_y[0:12000],data_y[12000:15000]



data = np.load('polution_dataSet.npy')
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
x_train,x_test,y_train,y_test = create_dataset(data,11)

model = Sequential()
model.add(LSTM(16,input_shape=(11, 8)))
model.add(Dense(16))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer="RMSprop")
hist = model.fit(x_train, y_train, epochs=65, batch_size=128, validation_split = 0.2, verbose=1)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Daily Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()



fig_Accuracy = plt.figure(figsize=(25, 6))
y_pred=model.predict(x_test)
real_daily_stat = []
predicted_daily_stat = []
for i in range(len(y_pred)):
    if(i%12==0):
        real_daily_stat.append(y_test[i])
        predicted_daily_stat.append(y_pred[i])
plt.plot(real_daily_stat, label= 'real pollution')
plt.plot(predicted_daily_stat, label = 'pollution prediction')
plt.title('Daily pollution prediction vs real pollution')
plt.legend()
plt.show()
model.save('daily_model.h5')