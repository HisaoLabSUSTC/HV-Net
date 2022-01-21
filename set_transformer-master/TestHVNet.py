import os
import sys

import numpy as np
import scipy.io as scio
import h5py
import torch
from models import *
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time

# CMD code: bash run_test.sh 7 model_whole_1.pth test_data_random_2.mat cuda
if __name__ == "__main__":
    path_dir = '/liaoweiduo/HV-Net-datasets'  # for ubuntu server
    # path_dir = '//10.20.2.245/datasets/HV-Net-datasets'  # for windows

    if len(sys.argv) is not 4:
        raise Exception('please specify model, test_data and device.')

    if sys.argv[3] == 'cpu':
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # model_file = 'model_50_50_100_1.pth'
    # model_file = 'model_tri_invertri_1.pth'
    # model_file = 'model_whole_1.pth'
    model_file = sys.argv[1]

    # test_file = 'test_data_tri_2.mat'
    # test_file = 'test_data_invtri_2.mat'
    # test_file = 'test_data_random_2.mat'
    test_file = sys.argv[2]

    save_file = f'result_{model_file[:-4]}_{test_file[:-4]}_{device}.mat'

    # Test
    path = os.path.join(path_dir, test_file)
    print(f'Test on: {test_file}')
    # data = scio.loadmat(path)
    data = h5py.File(path)

    def my_loss(output, pred):
        loss = torch.mean(abs(output - pred)/output)
        return loss

    ## loading process for scio
    # solutionset = torch.from_numpy(data.get('Data')).float()
    # hv = torch.from_numpy(data.get('HVval')).float()
    # hv = torch.reshape(hv, (hv.shape[0],1,1))

    ## loading process for h5py
    solutionset = torch.from_numpy(np.transpose(data.get('Data'))).float()  # [dataset_num, data_num, M]
    hv = torch.from_numpy(np.transpose(data.get('HVval'))).float()  # [dataset_num, 1]
    hv = torch.reshape(hv, (hv.shape[0], 1, 1))  # [dataset_num, num_outputs, dim_output]

    size = solutionset.shape[0]
    batch_size = 100

    dim_input = solutionset.shape[2]
    num_outputs = 1
    dim_output = 1

    model = DeepSet(device, dim_input, num_outputs, dim_output)
    model.load_state_dict(torch.load(os.path.join(path_dir, model_file)))
    model = model.to(device)
    print(f"Load model {model_file} done!")

    result = []
    dataloader = DataLoader(TensorDataset(solutionset, hv), batch_size=batch_size, num_workers=4)
    loss_fn = nn.MSELoss()

    num_batches = len(dataloader)
    model.eval()
    test_loss = []
    start_time = time.time()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            input, output = X.to(device), y.to(device)
            pred = model.forward_allow_nan(input)       # [bs, 1, 1]
            loss = my_loss(output, pred)
            # loss = 0
            # for i in range(batch_size):
            #     input, output = X[i:i+1], y[i:i+1]
            #     mask = ~torch.isnan(input[0,:,0])
            #     input = input[:,mask == True]
            #     pred = model(input)
            #     #loss = loss + loss_fn(pred,output)
            #     loss += abs(pred-output)/output
            # loss = loss/batch_size
            test_loss.append(loss.item())
            #pred = model(X)
            #test_loss += my_loss(y, pred)
            result.append(pred.cpu().detach().numpy())       # num_batches * [bs, 1, 1]
    test_loss = np.mean(test_loss)
    end_time = time.time()
    #correct /= size
    #print(f"Avg loss: {test_loss:>8f} \n")
    print(f'Avg Loss and Time Used: {[float(test_loss), end_time-start_time]}')
    scio.savemat(os.path.join(path_dir, save_file), {'result': np.concatenate(result, axis=0),
                                                     'loss': test_loss, 'time': end_time-start_time})
