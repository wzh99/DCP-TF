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
    pack_to_one()
