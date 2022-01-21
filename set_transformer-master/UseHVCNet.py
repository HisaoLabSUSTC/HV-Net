import scipy.io as scio
import torch
from models import *
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset


dim_input = 3
num_outputs = 1
dim_output = 1

model = DeepSetHVC(dim_input, num_outputs, dim_output)
model.load_state_dict(torch.load("model.pth"))

print("Load Done!")

path = 'testdata-HVC.mat'
data = scio.loadmat(path)

solutionset = torch.from_numpy(data.get('Data')).float()
hv = torch.from_numpy(data.get('HVCval')).float()
hv = torch.reshape(hv, (hv.shape[0],hv.shape[1],1))

index = 2
set = solutionset[index]
mask = ~torch.isnan(set)
set = set[mask[:, 0] == True]

print(model(set))
print(hv[index])