import h5py
import numpy as np


modelnet40_path = 'modelnet40/'


def pack_to_one():
    train_data = np.ndarray((0, 2048, 3))
    for i in range(5):
        f = h5py.File(modelnet40_path + 'ply_data_train%d.h5' % (i), 'r')
        train_data = np.concatenate([train_data, np.array(f['data'])])
    train_file = h5py.File('train.h5', 'a')
    train_file['data'] = train_data
    test_data = np.ndarray((0, 2048, 3))
    for i in range(2):
        f = h5py.File(modelnet40_path + 'ply_data_test%d.h5' % (i), 'r')
        test_data = np.concatenate([test_data, np.array(f['data'])])
    test_file = h5py.File('test.h5', 'a')
    test_file['data'] = test_data


if __name__ == "__main__":
    # import torch
    # wt = torch.load('weights/dcp_v1.t7', map_location=torch.device('cpu'))
    # for k, v in wt.items():
    #     print(k, v.size())
    h5 = h5py.File('dcp.h5', 'r')
    for k, g in h5['dgcnn']['dcp']['dgcnn'].items():
        # print(k, g)
        for k, v in g.items():
            print(k, v)
