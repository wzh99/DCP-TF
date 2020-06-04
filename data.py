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


def torch_to_tf(torch_path: str, tf_path: str):
    '''
    [dgcnn/dcp/dgcnn]
    [bn1/beta:0], ...gamma:0, ...moving_mean:0, ...moving_variance:0
    [conv1]: kernel:0 (1, 1, from, to)

    [emb_nn.bn1.weight], ...bias, ...running_mean, ...running_var]
    [emb_nn.conv1.weight] (to, from, 1, 1)
    '''

    # Open torch and tf weights
    import torch
    torch_wt = torch.load(torch_path, map_location=torch.device('cpu'))
    tf_wt = h5py.File(tf_path, mode='r+')

    # Transfer DGCNN weights
    dgcnn_group = tf_wt['dgcnn/dcp/dgcnn']
    for l in ['bn1', 'bn2', 'bn3', 'bn4', 'bn5']:
        grp = dgcnn_group[l]
        grp['beta:0'][...] = torch_wt['emb_nn.%s.bias' % l]
        grp['gamma:0'][...] = torch_wt['emb_nn.%s.weight' % l]
        grp['moving_mean:0'][...] = torch_wt['emb_nn.%s.running_mean' % l]
        grp['moving_variance:0'][...] = torch_wt['emb_nn.%s.running_var' % l]
    for l in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
        kernel = np.array(torch_wt['emb_nn.%s.weight' % l])
        kernel = np.transpose(kernel, axes=(3, 2, 1, 0))
        dgcnn_group[l]['kernel:0'][...] = kernel


if __name__ == "__main__":
    torch_to_tf('weights/dcp_v1.t7', 'dcp.h5')