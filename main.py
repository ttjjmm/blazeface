import torch

import numpy as np






if __name__ == '__main__':
    x = np.array([[1, 2, 3, 4], [6, 7, 3, 9], [1, 2, 3, 8], [6, 7, 8, 9]])
    print(x)
    z = x[:, [2, 3]] < 111
    print(z)
    indx = ~(z[:, 0] | z[:, 1])
    print(indx)
    res = x[indx, :]

    print(len(res))

