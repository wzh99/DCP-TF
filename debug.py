import model
import model_torch
import train
import tensorflow as tf
from tensorflow import keras
import torch
import numpy as np
import h5py


class OutputComparison:
    def __init__(self, torch_model: torch.nn.Module, tf_model: keras.Model):
        self.torch = torch_model
        self.tf = tf_model

    def compare_batch(self, x: np.ndarray, y: np.ndarray):
        print('Real Y:', y)
        loss = model.DCPLoss(None)

        # Run on PyTorch module
        x_torch = tf.transpose(x, perm=(0, 1, 3, 2))
        src, tgt = tf.split(x_torch, (1, 1), axis=1)
        src = torch.from_numpy(np.copy(np.squeeze(src)))
        tgt = torch.from_numpy(np.copy(np.squeeze(tgt)))
        self.torch.eval()
        R, t, _, _ = self.torch(src, tgt)
        t = np.expand_dims(t.detach().numpy(), 1)
        y_torch = np.concatenate([R.detach().numpy(), t], axis=1)
        print('Torch Output:', y)
        print('Torch Loss:', np.float32(loss.call(y, y_torch)))

        # Run on Tensorflow module
        y_tf = self.tf(x, training=False)
        print('TF Output:', y_tf)
        print('TF Loss:', np.float32(loss.call(y, y_tf)))


def compare_output():
    # Load data and initialize sequence
    data_file = h5py.File('train.h5', 'r')
    data = np.array(data_file['data'], dtype=np.float32)
    seq = train.DataSequence(data, 8)
    
    # Load TF model
    tf_model = model.DCP()
    tf_model(tf.zeros((2, 2, 2048, 3)))
    tf_model.load_weights('dcp_v2.h5')

    # Load torch model
    torch_model = model_torch.DCP(model_torch.Args())
    torch_wt = torch.load('weights/dcp_v2.t7',
                          map_location=torch.device('cpu'))
    # torch_model.load_state_dict(torch_wt)

    # Compare output
    cmp = OutputComparison(torch_model, tf_model)
    x, y = seq[0]
    cmp.compare_batch(x, y)


if __name__ == "__main__":
    compare_output()
