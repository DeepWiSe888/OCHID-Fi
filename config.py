import torch

train_on_gpu = torch.cuda.is_available()
if (train_on_gpu):
    print('Training on GPU.')
else:
    print('Training on CPU.')
DEVICE = torch.device('cuda' if train_on_gpu else 'cpu')

file_no_path = r'../data_no/'
file_oc_path = r'../data_oc/'
x_no_path = file_no_path+'rf/'
x_oc_path = file_oc_path+'rf/'
y_no_path = file_no_path +'label/'
y_oc_path = file_oc_path +'label/'
model_path = './model/'


