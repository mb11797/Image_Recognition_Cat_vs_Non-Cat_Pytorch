from lr_utils import load_dataset
import torch
import cv2

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

model = torch.load('model_best.pt')



test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1) / 255

X_test = torch.from_numpy(test_x_flatten).float()



y_test = torch.from_numpy(test_set_y.T).long()
y_test = y_test.view(50)

y_pred = model(X_test)
for i in range(50) :
    if y_pred[i][0] > y_pred[i][1]:
        print(y_pred[i], 'Not Cat', '\t', y_test[i], '\t', classes[test_set_y[0, i]].decode("utf-8"))
    elif y_pred[i][0] < y_pred[i][1]:
        print(y_pred[i], 'Cat', '\t', y_test[i], '\t', classes[test_set_y[0, i]].decode("utf-8"))


# cv2.imshow('img0', train_set_x_orig[11])
# cv2.waitKey(0)
# cv2.destroyAllWindows()










