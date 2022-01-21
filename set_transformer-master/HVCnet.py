import scipy.io as scio
import torch
from models import *
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random

path = 'testdata-HVC.mat'
data = scio.loadmat(path)

solutionset = torch.from_numpy(data.get('Data')).float()
hv = torch.from_numpy(data.get('HVCval')).float()
hv = torch.reshape(hv, (hv.shape[0],hv.shape[1],1))

#index = [i for i in range(len(solutionset))]
#random.shuffle(index)
#hv = [hv[i] for i in index]
#solutionset = [solutionset[i] for i in index]

dim_input = solutionset.shape[2]
num_outputs = 1
dim_output = 1


settransformer = DeepSetHVC(dim_input, num_outputs, dim_output)
#settransformer = SetTransformer(dim_input, num_outputs, dim_output)
#settransformer.load_state_dict(torch.load("model.pth"))

optimizer = torch.optim.Adam(settransformer.parameters(), lr=1e-4)
#optimizer = torch.optim.SGD(settransformer.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 10
dataloader = DataLoader(TensorDataset(solutionset, hv), batch_size=batch_size, shuffle=True)

epochs = 5000
# Train
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    size = solutionset.shape[0]
    #deepset.train()
    settransformer.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        loss = 0
        for i in range(batch_size):
            input, output = X[i:i+1], y[i:i+1]
            mask = ~torch.isnan(input)
            input = input[:, mask[0, :, 0] == True]
            output = output[:, mask[0, :, 0] == True]
            pred = settransformer(input)
            #loss += loss_fn(pred, output)
            loss += loss_fn(torch.log(pred), torch.log(output))
        #loss = loss/sum(y)
        loss = loss/batch_size
        #pred = settransformer(X)
        #loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
print("Train Done!")



torch.save(settransformer.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")


#model = DeepSet(dim_input, num_outputs, dim_output)
#model.load_state_dict(torch.load("model.pth"))

#print("Load Done!")