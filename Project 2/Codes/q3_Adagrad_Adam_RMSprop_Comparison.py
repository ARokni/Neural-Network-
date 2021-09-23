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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import time

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
validation_hist_list = []
legend = ["adam","RMSprop","Adagrad"]
test_loss_list = []
training_time = []
for i in range(3):
  test_loss_list.append([])
  training_time.append([])
  validation_hist_list.append([])
for i in range(10):
  for j in range(3):
    model = Sequential()
    model.add(SimpleRNN(16,input_shape=(11, 8)))
    model.add(Dense(16))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss="mean_absolute_error", optimizer=legend[j])
    t = time.time()
    hist = model.fit(x_train, y_train, epochs=60, batch_size=64, validation_split = 0.2, verbose=0)
    training_time[j].append(time.time() - t)
    validation_hist_list[j].append(hist.history['val_loss'])
    test_loss_list[j].append(mean_absolute_error(y_test,model.predict(x_test)))
    print(i*3 +j)
    
    
test_loss_mean = []
test_loss_std = []
for i in range(3):
  test_loss_mean.append(np.mean(test_loss_list[i]))
  test_loss_std.append(np.std(test_loss_list[i])/np.sqrt(10))
plt.errorbar(legend,test_loss_mean,yerr=test_loss_std)
plt.xticks(rotation=0)
plt.title("mean absolute error on test data")
plt.show()


training_time_mean = []
training_time_std = []
for i in range(3):
  training_time_mean.append(np.mean(training_time[i]))
  training_time_std.append(np.std(training_time[i])/np.sqrt(10))
plt.errorbar(legend,training_time_mean,yerr=training_time_std)
plt.xticks(rotation=0)
plt.title("training time")
plt.show()


validation_plot = []
for i in range(3):
  x = [sum(x)/10 for x in zip(*validation_hist_list[i])]
  validation_plot.append(x)
  plt.plot(validation_plot[i], label = legend[i])
plt.title('Model validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

