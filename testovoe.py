import nasdaqdatalink
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from extraction import window, transform_pred1, transform_pred2

np.random.seed(1234)
data_width = 500
label_width = 50

def train_val_test_split_and_reshape(data, train_frac, test_frac):
  n = len(data)
  train_data = data[0:int(n*train_frac)]
  train_data = np.asarray(train_data).reshape((len(train_data), len(train_data[0]), 1))
  val_data = data[int(n*train_frac):int(n*(1-test_frac))]
  val_data = np.asarray(val_data).reshape((len(val_data), len(val_data[0]), 1))
  test_data = data[int(n*(1-test_frac)):]
  test_data = np.asarray(test_data).reshape((len(test_data), len(test_data[0]), 1))
  return (train_data, val_data, test_data)

# WTI Crude Oil price from the US Department of Energy
mydata = nasdaqdatalink.get("EIA/PET_RWTC_D")

# Убрал отрицительное значение
mydata.iloc[np.where(mydata<0)[0], :] = 9.00

# Убрал даты - оставил только цены
mydata.reset_index(inplace=True)
mydata = mydata["Value"]

# Стандартизация
data_mean = mydata.mean()
data_std = mydata.std()
mydata = (mydata - data_mean) / data_std

# Генерация данных
X, y = window(mydata, data_width, label_width)

train_X, val_X, test_X = train_val_test_split_and_reshape(X, 0.7, 0.1)
train_y, val_y, test_y = train_val_test_split_and_reshape(y, 0.7, 0.1)

# Linear model
Linear_model = Sequential()
Linear_model.add(Dense(units=50, activation='sigmoid', input_shape=(train_X.shape[1], )))
Linear_model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=[MeanAbsoluteError()])
Linear_model.summary()

# Dense model
Dense_model = Sequential()
Dense_model.add(Dense(512, activation='relu', input_shape=(train_X.shape[1], )))
Dense_model.add(Dense(50, activation='sigmoid'))

Dense_model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=[MeanAbsoluteError()])
Dense_model.summary()

# LSTM network (easy)
eLSTM_model = Sequential()
eLSTM_model.add(LSTM(
         input_shape=(train_X.shape[1], train_X.shape[2]),
         units=32,
         return_sequences=False))

eLSTM_model.add(Dense(units=50, activation='sigmoid'))
eLSTM_model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=[MeanAbsoluteError()])
eLSTM_model.summary()

# LSTM network (complex)
cLSTM_model = Sequential()
cLSTM_model.add(LSTM(
         input_shape=(train_X.shape[1], train_X.shape[2]),
         units=100,
         return_sequences=True))
cLSTM_model.add(Dropout(0.2))

cLSTM_model.add(LSTM(
          units=50,
          return_sequences=False))
cLSTM_model.add(Dropout(0.2))

cLSTM_model.add(Dense(units=50, activation='sigmoid'))
cLSTM_model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=[MeanAbsoluteError()])
cLSTM_model.summary()

# Обучение
history_elstm = eLSTM_model.fit(train_X, train_y, epochs=20, batch_size=200, validation_data=(val_X, val_y), verbose=1,
                    callbacks = [EarlyStopping(monitor='val_loss', patience=2, mode='min')])
history_clstm = cLSTM_model.fit(train_X, train_y, epochs=20, batch_size=200, validation_data=(val_X, val_y), verbose=1,
                    callbacks = [EarlyStopping(monitor='val_loss', patience=1, mode='min')])
history_linear = Linear_model.fit(train_X, train_y, epochs=20, batch_size=200, validation_data=(val_X, val_y), verbose=1,
                    callbacks = [EarlyStopping(monitor='val_loss', patience=2, mode='min')])
history_dense = Dense_model.fit(train_X, train_y, epochs=20, batch_size=200, validation_data=(val_X, val_y), verbose=1,
                    callbacks = [EarlyStopping(monitor='val_loss', patience=2, mode='min')])

# Предасказания
pred_e = eLSTM_model.predict(test_X)
pred_c = cLSTM_model.predict(test_X)
pred_d = Dense_model.predict(test_X)
pred_l = Linear_model.predict(test_X)

# Форматирование предсказаний
transformed_preds = []
for pred in [pred_e, pred_c, pred_d, pred_l]:
  transformed_preds.append(transform_pred1(pred))

# Сравнение предсказаний для конкретного окна
# plt.plot(pred_e[-1] * data_std + data_mean, label='pred')
# plt.plot(test_y[-1] * data_std + data_mean, label='true')
# plt.legend()
# plt.show()

# Качество моделей на тестовых данных
for model in [eLSTM_model, cLSTM_model, Linear_model, Dense_model]:
  model.evaluate(test_X, test_y)

# Предсказания cLSTM-модели
pred_data = transformed_preds[1]
plt.figure(figsize=(15,8))
plt.plot(mydata[:-len(pred_data)-len(val_X)] * data_std + data_mean, label="train", c='0.8')
plt.plot(mydata[-len(pred_data)-len(val_X):-len(pred_data)] * data_std + data_mean, label="val", c='gray')
plt.plot(pd.Series(mydata[-len(pred_data):] * data_std + data_mean), label="test", c='#008000')
plt.plot((pd.Series(pred_data, [len(mydata)-len(pred_data)+i for i in range(len(pred_data))]) * data_std + data_mean), label="pred", c='#00FF00')
plt.legend()
plt.show()
