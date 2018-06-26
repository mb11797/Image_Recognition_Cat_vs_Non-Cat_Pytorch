from lr_utils import load_dataset
import torch
from sklearn.preprocessing import OneHotEncoder

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

print(type(train_set_x_orig))
print('train_set_x_orig.shape : ', train_set_x_orig.shape)
print('train_set_y.shape[0] : ', train_set_y.shape[0])
print('train_set_y.shape : ', train_set_y.shape)

#
# train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1) / 255
# test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1) / 255
#
# print('train_x_flatten.shape : ', train_x_flatten.shape)
# print('test_x_flatten.shape : ', test_x_flatten.shape)
# # print(train_set_y.shape)
#
# N = train_set_x_orig.shape[0]
# D_in = train_set_x_orig.shape[1] * train_set_x_orig.shape[2] * train_set_x_orig.shape[3]
# H = 100
# D_out = train_set_y.shape[0]
#
# X = torch.from_numpy(train_x_flatten).float()
# y = torch.from_numpy(train_set_y)
#
# print('y shape : ', y.shape)
#
# model = torch.nn.Sequential(
#     torch.nn.Linear(D_in, H),
#     torch.nn.ReLU(),
#     torch.nn.Sigmoid(),
#     # torch.nn.Linear(H, D_out)
# )


train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1) / 255
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1) / 255

print('train_x_flatten.shape : ', train_x_flatten.shape)
print('test_x_flatten.shape : ', test_x_flatten.shape)
# print(train_set_y.shape)

N = 209
D_in = 12288
H = 10
D_out = 2

y = train_set_y

# oneHot = OneHotEncoder(categorical_features=[0])
# y = oneHot.fit_transform(y)
# # print(y)

X = torch.from_numpy(train_x_flatten).float()

# y = torch.from_numpy(y.T).long()
# print (y.shape)

y = torch.empty(209, dtype=torch.long).random_(1)



print('y shape : ', y.shape)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),

    torch.nn.Linear(H, D_out),
    torch.nn.Sigmoid(),
)


# model = model.double()

loss_fn = torch.nn.CrossEntropyLoss()
# loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 0.0075

for t in range(500):
    y_pred = model(X)
    print ("break")

    print (y_pred)


    print('y_pred shape : ', y_pred.shape)


    # for i in range(y_pred.shape[0]):
    #     if y_pred[i] < 0.5:
    #         y_pred[i] = 0
    #     else :
    #         y_pred[i] = 1

    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    model.zero_grad()

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad






