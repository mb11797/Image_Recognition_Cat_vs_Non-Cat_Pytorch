from lr_utils import load_dataset
import torch


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1) / 255
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1) / 255



N = 209
D_in = 12288
H = 10
D_out = 1

X = torch.from_numpy(train_x_flatten).float()

y = torch.LongTensor(train_set_y.T)
print (y.shape)
print(y)
y = y.view(209)
print (y.shape)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.Sigmoid(),
)
# Loss and Optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(X)

learning_rate = 0.0075
for t in range(500):
    y_pred = model(X)

    # print(y_pred)

    loss = loss_fn(y_pred, y)
    print(t, loss.item())


