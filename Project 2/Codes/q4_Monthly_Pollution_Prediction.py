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
def create_dataset(dataset):
    hour = np.random.randint(0,24)
    day = np.random.randint(0,7)
    tmp=0
    tmp_list = []
    while(tmp + day*24 + hour<len(dataset)):
        tmp_list.append(dataset[tmp + day*24 + hour,:])
        tmp +=7*24
    data_x, data_y = [], []
    for i in range(len(tmp_list)-3):
        data_x.append(tmp_list[i:i+3])
        data_y.append(tmp_list[i+3][0])
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    return data_x[0:220,:],data_x[220:,:],data_y[0:220],data_y[220:]


data = np.load('polution_dataSet.npy')
#scaler = StandardScaler()
#scaler.fit(data)
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
x_train,x_test,y_train,y_test = create_dataset(data)

model = Sequential()
model.add(SimpleRNN(20,recurrent_dropout=0.05,kernel_regularizer=l2(0.00001), input_shape=(3, 8), return_sequences=True))
model.add(SimpleRNN(20,recurrent_dropout=0.05,kernel_regularizer=l2(0.00001),return_sequences=True))
model.add(SimpleRNN(20,recurrent_dropout=0.05,kernel_regularizer=l2(0.00001)))
model.add(Dense(20))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer="RMSprop")

hist = model.fit(x_train, y_train, epochs=140, batch_size=8, verbose=1,validation_data=(x_test, y_test))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


fig_Accuracy = plt.figure(figsize=(25, 4))
y_pred=model.predict(x_test)
plt.plot(y_test)
plt.plot(y_pred)
plt.show()
fig_Accuracy = plt.figure(figsize=(25, 4))
y_pred=model.predict(x_train)
plt.plot(y_train)
plt.plot(y_pred)
plt.show()

model.save('monthly_model.h5')