import numpy as np


def hv_mc(popobj, r=1.1, sample_num=5000):
    m = np.size(popobj, 1)
    n = np.size(popobj, 0)
    max_value = np.ones((1, m)) * r
    min_value = np.min(popobj, axis=0)
    samples = np.random.rand(sample_num, m) * (max_value-min_value) + min_value
    for i in range(n):
        domi = np.ones(np.size(samples, 0), dtype=bool)
        c = 0
        while c < m and np.any(domi):
            domi = domi & (popobj[i, c] <= samples[:, c])
            c = c + 1
        samples = samples[~domi, :]

    score = np.prod(max_value-min_value) * (1-np.size(samples, 0)/sample_num)

    return score


# Simple Test. Notice: The correctness is not tested!!!
# popobj = np.array([[1.0, 1.0, 1.0],
#                    [0.3, 0.4, 0.5]])
# print(hv_mc(popobj))
