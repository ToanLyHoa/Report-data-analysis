import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
import numpy as np
from Model_train import ulti
torch.set_default_tensor_type('torch.cuda.FloatTensor')


class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length


        self.batchnorm1 = nn.BatchNorm1d(input_size, affine = True)

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1

        self.batchnorm2 = nn.BatchNorm1d(128, affine = True)

        self.relu = nn.ReLU()

        self.fc = nn.Linear(128, num_classes) #fully connected last layer
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        
        x = torch.transpose(x, 1, 2)
        x = self.batchnorm1(x)
        x = torch.transpose(x, 1, 2)

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.batchnorm2(out)
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out


X = np.loadtxt("../../Data/Data/data_train/arrayX_train_30_30_30_7_days_pred_7_days_2020.txt", delimiter=",")
Y = np.loadtxt("../../Data/Data/data_train/arrayY_train_30_30_30_7_days_pred_7_days_2020.txt", delimiter=",")

# Giữ lại một số sku để test
sku_test = 103
sku_train = round(X.shape[0]/329.0 - sku_test)

day_temp = 329


# Giữ lại một số sku để test
sku_test = 103
sku_val = 100
sku_train = round(X.shape[0]/329.0 - sku_test - sku_val)

day_temp = 329


# split data into train and test sets
seed = 7
test_size = 0.2
X_train, y_train= X[:sku_train*day_temp], Y[:sku_train*day_temp]
X_val, y_val = X[sku_train*day_temp:(sku_train + sku_val)*day_temp], Y[sku_train*day_temp:(sku_train + sku_val)*day_temp]
X_test, y_test = X[(sku_train + sku_val)*day_temp:], Y[(sku_train + sku_val)*day_temp:]

X_train = X_train[:, :90]
X_val = X_val[:, :90]
X_test = X_test[:, :90]
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)



# Chuẩn bị data dể train và test
# X_train_tensors = Variable(torch.Tensor(X_train.reshape(-1,3,30).transpose(0,2,1)))
# X_val_tensors = Variable(torch.Tensor(X_val.reshape(-1,3,30).transpose(0,2,1)))
# X_test_tensors = Variable(torch.Tensor(X_test.reshape(-1,3,30).transpose(0,2,1)))

X_train_tensors = Variable(torch.Tensor(X_train.reshape(-1,3,30)))
X_val_tensors = Variable(torch.Tensor(X_val.reshape(-1,3,30)))
X_test_tensors = Variable(torch.Tensor(X_test.reshape(-1,3,30)))

y_train_tensors = Variable(torch.Tensor(y_train))
y_val_tensors = Variable(torch.Tensor(y_val))
y_test_tensors = Variable(torch.Tensor(y_test)) 


num_epochs = 1000 #1000 epochs
learning_rate = 0.001 #0.001 lr

input_size = 30 #number of features
hidden_size = 128 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

num_classes = 7 #number of output classes 

batch_size = 128


lstm = LSTM(num_classes = num_classes, input_size = input_size,
                    hidden_size = hidden_size, num_layers = num_layers, seq_length = 30)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate) 



import time
start_time_total = time.time()


for epoch in range(num_epochs):
    # Đếm giờ train cho mỗi epoch
    start_time = time.time()

    # X is a torch Variable
    permutation = torch.randperm(X_train_tensors.size()[0])

    loss_train = 0
    for i in range(0,X_train_tensors.size()[0], batch_size):
            optimizer.zero_grad()         #caluclate the gradient, manually setting to 0

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_tensors[indices], y_train_tensors[indices]

            # in case you wanted a semi-full example
            outputs = lstm.forward(batch_x)
            loss = criterion(outputs,batch_y)

            loss_train += loss.item() * batch_x.size(0)

            loss.backward() #calculates the loss of the loss function
            optimizer.step() #improve from loss, i.e backprop

    lstm.eval()
    loss_val = 0
    with torch.no_grad():
        permutation_val = torch.randperm(X_val_tensors.size()[0])

        for i in range(0,X_val_tensors.size()[0], batch_size):
            optimizer.zero_grad()         #caluclate the gradient, manually setting to 0

            indices_val = permutation_val[i:i+batch_size]
            batch_x_val, batch_y_val = X_val_tensors[indices_val], y_val_tensors[indices_val]

            # in case you wanted a semi-full example
            outputs_val = lstm.forward(batch_x_val)
            loss_val_temp = criterion(outputs_val, batch_y_val)

            loss_val += loss_val_temp.item() * batch_x_val.size(0)

    lstm.train()


    print("Epoch: %d, loss: %1.5f" % (epoch, loss_train/X_train_tensors.size(0))) 
    print("Epoch: %d, loss_val: %1.5f" % (epoch, loss_val/X_val_tensors.size(0))) 
    print("Epoch: %d, --- %s seconds ---" % (epoch, time.time() - start_time))

    if (epoch + 1)%100 == 0:
        ulti.save_checkpoint(lstm, optimizer, 
                    f'../../Data/Data/model_train/lstm_transpose_batchnorm_30_30_30_hidden_128_pred_7_epoch_{epoch}', epoch=epoch)


print("--- %s seconds ---" % (time.time() - start_time_total))



















