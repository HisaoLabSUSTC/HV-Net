import os
import sys

import numpy as np
import scipy.io as scio
import h5py
import time

from MC import hv_mc
from R2HV import hv_r2

# CMD code: python TestMCR2.py 3
if __name__ == "__main__":
    path_dir = '/liaoweiduo/HV-Net-datasets'  # for ubuntu server
    # path_dir = '//10.20.2.245/datasets/HV-Net-datasets'  # for windows

    if len(sys.argv) == 2:
        M = int(sys.argv[1])
    else:
        M = 3

    # 1-19 seeds
    test_files = [f'test_data_M{M}_{seed}.mat' for seed in range(1, 20)]

    for test_file in test_files:
        save_file = f'result_MC_R2_{test_file[:-4]}.mat'

        # Test
        path = os.path.join(path_dir, test_file)
        print(f'Test on: {test_file}')
        # data = scio.loadmat(path)
        data = h5py.File(path)

        def my_loss(output, pred):
            loss = np.mean(np.abs(output - pred)/output)
            return loss

        ## loading process for scio
        # solutionset = torch.from_numpy(data.get('Data')).float()
        # hv = torch.from_numpy(data.get('HVval')).float()
        # hv = torch.reshape(hv, (hv.shape[0],1,1))

        ## loading process for h5py
        solutionset = np.transpose(data.get('Data'))  # [dataset_num, data_num, M]
        hv = np.transpose(data.get('HVval')) # [dataset_num, 1]
        hv = hv.reshape((hv.shape[0]))  # [dataset_num]

        L, N, M = solutionset.shape
        sampleNum = np.arange(100, 2001, 100)   # [20,]   [100, 200, ..., 2000]
        lineNum = np.arange(10, 201, 10)        # [20,]   [10, 20, ..., 200]

        Time_MC = np.zeros(len(sampleNum))
        Loss_MC = np.zeros(len(sampleNum))
        Time_R2 = np.zeros(len(lineNum))
        Loss_R2 = np.zeros(len(lineNum))

        for k in range(len(sampleNum)):
            print(f'MC {k}')
            loss = []
            start_time = time.time()
            for i in range(L):
                pop = solutionset[i]        # [N, M]
                mask = ~np.isnan(pop[:, 0])  # [100]
                pop = pop[mask == True, :]  # [30, 3]
                hvmc = hv_mc(pop, 1, sampleNum[k])
                loss.append(my_loss(hv[i], hvmc))
            loss = np.mean(loss)
            end_time = time.time()

            Loss_MC[k] = loss
            Time_MC[k] = end_time - start_time
            print(f'loss {loss}, time {end_time - start_time}')

        for k in range(len(lineNum)):
            print(f'R2 {k}')
            loss = []
            start_time = time.time()
            for i in range(L):
                pop = solutionset[i]  # [N, M]
                mask = ~np.isnan(pop[:, 0])  # [100]
                pop = pop[mask == True, :]  # [30, 3]
                hvr2 = hv_r2(pop, 1, lineNum[k])
                loss.append(my_loss(hv[i], hvr2))
            loss = np.mean(loss)
            end_time = time.time()

            Loss_R2[k] = loss
            Time_R2[k] = end_time - start_time
            print(f'loss {loss}, time {end_time - start_time}')

        # save
        scio.savemat(os.path.join(path_dir, save_file), {'Loss': Loss_MC,  'Time': Time_MC,
                                                         'Loss1': Loss_R2, 'Time1': Time_R2})
