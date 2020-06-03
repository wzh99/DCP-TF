import h5py
import numpy as np
import tensorflow as tf
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
from tensorflow import keras
from tqdm import trange
from datetime import datetime
import os

import model


epochs = 100
batch_size = 16
model_path = 'dcp.h5'


class DataSequence(keras.utils.Sequence):
    def __init__(self, data: tf.Tensor, batch_size: int):
        super().__init__()
        self.data = data
        self.len = len(data)
        self.batch_size = batch_size

    def __len__(self):
        return (self.len - 1) // self.batch_size + 1

    def __getitem__(self, idx: int):
        # Fetch data for this batch
        batch_slice = slice(idx * self.batch_size,
                            min((idx + 1) * self.batch_size, self.len))
        src = data[batch_slice]
        batch_len = len(src)

        # Compute true transformation
        euler_angles = np.random.uniform(
            low=-np.pi, high=np.pi, size=(batch_len, 3)).astype(np.float32)
        R_true = rotation_matrix_3d.from_euler(euler_angles)
        R_expand = tf.expand_dims(R_true, 1)
        t_true = np.random.uniform(
            low=-0.5, high=0.5, size=(batch_len, 3)).astype(np.float32)
        t_expand = tf.expand_dims(t_true, 1)

        # Output batch
        tgt = rotation_matrix_3d.rotate(src, R_expand) + t_expand
        src = tf.expand_dims(src, axis=1)
        tgt = tf.expand_dims(tgt, axis=1)
        x = tf.concat([src, tgt], 1)
        y_true = tf.concat([R_true, t_expand], 1)

        return x, y_true

    def on_epoch_end(self):
        np.random.shuffle(self.data)


def train():
    # Load data
    file = h5py.File('train.h5', 'r')
    data = np.array(file['data'], dtype=np.float32)
    seq = DataSequence(data, batch_size)

    # Build model
    dcp = model.DCP()
    dcp.compile(optimizer='adam', loss=model.DCPLoss(dcp))
    dcp(np.zeros((batch_size, 2, 2048, 3)))
    if os.path.exists(model_path):
        print('Model weights found.')
        dcp.load_weights(model_path)

    # Setup summary writer
    log_dir = 'logs/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/train'
    writer = tf.summary.create_file_writer(log_dir)

    # Training loop
    num_batches = len(seq)
    best_loss = float('inf')

    for epoch_idx in range(epochs):
        # Train on batches
        losses = np.zeros((num_batches), dtype=np.float32)
        # progress = trange(num_batches)
        # for batch_idx in progress:
        for batch_idx in range(num_batches):
            x, y = seq[batch_idx]
            batch_loss = dcp.train_on_batch(x, y)
            losses[batch_idx] = batch_loss
            # progress.set_description('Loss: %s' % (
            # np.sum(losses) / (batch_idx + 1)))
        epoch_loss = np.mean(losses)

        # Write epoch loss
        with writer.as_default():
            tf.summary.scalar('loss', epoch_loss, step=epoch_idx)

        # Save if lower loss is achieved
        if epoch_loss < best_loss:
            dcp.save_weights(model_path)
            best_loss = epoch_loss

        # Update data sequence
        seq.on_epoch_end()


if __name__ == "__main__":
    train()