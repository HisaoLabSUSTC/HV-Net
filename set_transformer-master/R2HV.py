import numpy as np
from math import gamma

def uniform_vector(sample_num, m):
    mu = np.zeros(m)
    sigma = np.eye(m, m)
    sample_r = np.random.multivariate_normal(mu, sigma, sample_num)
    sample_v = np.abs(sample_r/np.sqrt(np.sum(sample_r**2, axis=1)).reshape(-1, 1))
    # print(sample_v)
    return sample_v


def hv_r2(popobj, r=1.1, sample_num=5000):
    m = np.size(popobj, 1)
    n = np.size(popobj, 0)
    max_value = np.ones((1, m)) * r
    min_value = np.min(popobj, axis=0)
    samples = uniform_vector(sample_num, m)

    y = 0
    for j in range(sample_num):
        temp = np.abs(popobj - max_value)/samples[j,:]
        x = np.max(np.min(temp, axis=1))
        y = y + x ** m

    score = y * np.pi ** (m / 2) / (m * sample_num * 2 ** (m - 1) * gamma(m / 2))

    return score


# Simple Test. Notice: The correctness is not tested!!!
# popobj = np.array([[1.0, 1.0, 1.0],
#                    [0.3, 0.4, 0.5]])
# print(hv_r2(popobj))
