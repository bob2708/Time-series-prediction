#import pandas as pd
import numpy as np

# Генерация данных
def window(data, data_width, label_width):
  X = []
  y = []
  for i in range(data_width, len(data)-label_width):
    X.append(data.values[i-data_width:i])
    y.append(data.values[i:i+label_width])
  return (X, y)

# Форматирование предсказаний вариант 1
def transform_pred1(pred):
  pred_data = []
  for i, val in enumerate(pred):
    if i==0: 
      pred_data = val
      continue
    else: pred_data = np.append(pred_data,val[-1])
  return pred_data

# Форматирование предсказаний вариант 2
def transform_pred2(pred):
  pred_data = []
  for i in range(len(pred)//50+1):
    pred_data.extend(pred[i*50])
  pred_data.extend(pred[-1][-len(pred)%50+1:])
  print(len(pred_data))
  return pred_data