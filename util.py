import h5py
import numpy as np
import torch
from collections import OrderedDict

import model
import train


modelnet40_path = 'modelnet40/'


def pack_to_one():
    train_data = np.ndarray((0, 2048, 3), dtype=np.float32)
    for i in range(5):
        f = h5py.File(modelnet40_path + 'ply_data_train%d.h5' % (i), 'r')
        train_data = np.concatenate([train_data, np.array(f['data'])])
    train_file = h5py.File('train.h5', 'a')
    train_file['data'] = train_data
    test_data = np.ndarray((0, 2048, 3), dtype=np.float32)
    for i in range(2):
        f = h5py.File(modelnet40_path + 'ply_data_test%d.h5' % (i), 'r')
        test_data = np.concatenate([test_data, np.array(f['data'])])
    test_file = h5py.File('test.h5', 'a')
    test_file['data'] = test_data


def transfer_from_torch(torch_path: str):
    # Save untrained weights
    dcp = model.DCP()
    dcp(np.zeros((2, 2, 2048, 3), dtype=np.float32))
    dcp.save_weights(train.model_path)

    # Open torch and tf weights
    torch_wt = torch.load(torch_path, map_location=torch.device('cpu'))
    # for k, v in torch_wt.items():
    #     print(k, v.size())
    tf_wt = h5py.File(train.model_path, mode='r+')

    # Transfer DGCNN weights
    dgcnn_grp = tf_wt['dgcnn/dcp/dgcnn']
    for l in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
        kernel = np.array(torch_wt['emb_nn.%s.weight' % l])
        kernel = np.transpose(kernel, axes=(3, 2, 1, 0))
        dgcnn_grp[l]['kernel:0'][...] = kernel

    # Transfer Transformer weights
    trans_grp = tf_wt['transformer/dcp/transformer']
    dec_grp = trans_grp['decoder/d0']
    torch_dec = 'pointer.model.decoder.layers.0'
    _transfer_mha(dec_grp, 'd0_mha1', torch_wt, torch_dec + '.self_attn')
    _transfer_mha(dec_grp, 'd0_mha2', torch_wt, torch_dec + '.src_attn')
    _transfer_ffn(dec_grp, 'd0_ffn', torch_wt, torch_dec + '.feed_forward')
    enc_grp = trans_grp['encoder/e0']
    torch_enc = 'pointer.model.encoder.layers.0'
    _transfer_mha(enc_grp, 'e0_mha', torch_wt, torch_enc + '.self_attn')
    _transfer_ffn(enc_grp, 'e0_ffn', torch_wt, torch_enc + '.feed_forward')

    # Load weights to validate
    dcp.load_weights(train.model_path)


def _transfer_mha(tf_grp: h5py.Group, tf_layer: str, torch_wt: OrderedDict,
                  torch_layer: str):
    wt_corres = ['q', 'k', 'v', 'd']
    for i in range(len(wt_corres)):
        tf_grp['%s/%s_%s/kernel:0' % (tf_layer, tf_layer, wt_corres[i])
               ][...] = np.transpose(
                   torch_wt['%s.linears.%d.weight' % (torch_layer, i)],
                   axes=(1, 0))
        tf_grp['%s/%s_%s/bias:0' % (tf_layer, tf_layer, wt_corres[i])
               ][...] = torch_wt['%s.linears.%d.bias' % (torch_layer, i)]


def _transfer_ffn(tf_grp: h5py.Group, tf_layer: str, torch_wt: OrderedDict,
                  torch_layer: str):
    for i in [1, 2]:
        tf_grp['%s/%s_d%d/kernel:0' % (tf_layer, tf_layer, i)
               ][...] = np.transpose(
                   torch_wt['%s.w_%d.weight' % (torch_layer, i)],
                   axes=(1, 0))
        tf_grp['%s/%s_d%d/bias:0' % (tf_layer, tf_layer, i)
               ][...] = torch_wt['%s.w_%d.bias' % (torch_layer, i)]


if __name__ == "__main__":
    transfer_from_torch('weights/dcp_v2.t7')
