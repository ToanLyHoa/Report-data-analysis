from Data_train import epoch 
from Data_train import data 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from Data_train import epoch 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
import numpy as np

create_epoch_date = epoch.create_epoch_date

model = XGBRegressor(tree_method='gpu_hist', gpu_id = 0, max_depth =25)


create_epoch_date = epoch.create_epoch_date

# # create_epoch_hour()

# # model = XGBRegressor(tree_method='gpu_hist', gpu_id=0)
# X, Y = create_epoch_date(2020)

# print('Load xong dữ liệu')

# with open("../../Data/Data/data_train/arrayX_train_30_30_30_7_days_pred_7_days_2020.txt", "a") as arrayX:
#           np.savetxt(arrayX, X, delimiter=",", fmt='%.4f')

# with open("../../Data/Data/data_train/arrayY_train_30_30_30_7_days_pred_7_days_2020.txt", "a") as arrayY:
#           np.savetxt(arrayY, Y, delimiter=",", fmt='%.4f')

# print('Hoàn thành lưu vào Data')

# # split data into train and test sets
# seed = 7
# test_size = 0.2
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# model.fit(X_train, y_train)
# print(model)

X = np.loadtxt("../../Data/Data/data_train/arrayX_train_30_30_30_7_days_pred_7_days_2020.txt", delimiter=",")
Y = np.loadtxt("../../Data/Data/data_train/arrayY_train_30_30_30_7_days_pred_7_days_2020.txt", delimiter=",")

# Giữ lại một số sku để test
sku_test = 103
sku_train = round(X.shape[0]/329.0 - sku_test)

day_temp = 329
seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test = X[:sku_train*day_temp], X[sku_train*day_temp:], Y[:sku_train*day_temp], Y[sku_train*day_temp:]
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model.fit(X_train, y_train)
print(model)

model.save_model('../../Data/Data/data_train/model_train_30_30_30_7_days_pred_7_days_2020')