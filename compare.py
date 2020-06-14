import numpy as np
import tensorflow as tf
import h5py
import time

import model
import train


# Model indices in test dataset
test_size = 100
# Seed for MT19937 bit generator
rng_seed = 1


def test_dcp():
    '''
    Test performance of DCP on the first 100 models of test dataset.
    '''
    # Load data
    data_file = h5py.File('test.h5', 'r')
    data = data_file['data'][:test_size]
    data = np.array(data, dtype=np.float32)

    # Build model
    dcp = model.DCP()
    dcp.compile(loss=model.DCPLoss(dcp))
    dcp(tf.zeros((2, 2, 2048, 3)))
    dcp.load_weights(train.model_path, by_name=True, skip_mismatch=True)

    # Test on selected data
    np.random.seed(rng_seed)
    seq = train.DataSequence(data, 10)
    losses = np.ndarray((len(seq),), dtype=np.float32)
    total_time = 0
    print('Testing DCP')
    for i in range(len(seq)):
        x, y = seq[i]
        start = time.time()
        losses[i] = dcp.test_on_batch(x, y)
        total_time += time.time() - start
    print('Loss:', np.mean(losses))
    print('Time: %f s' % (total_time / test_size))


if __name__ == "__main__":
    test_dcp()
