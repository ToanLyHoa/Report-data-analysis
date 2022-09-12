from Data_train import epoch 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor

create_batch = epoch.create_epoch

model = XGBRegressor(tree_method='gpu_hist', gpu_id=0)
X, Y = create_batch()


# split data into train and test sets
seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# model.fit(X_train, y_train)
# print(model)