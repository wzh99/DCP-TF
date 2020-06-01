import tensorflow as tf
from tensorflow import keras
from typing import Optional, Tuple


class DCP(keras.Model):
    def __init__(self):
        super().__init__()
        self.dgcnn = DGCNN()
        self.transformer = Transformer(1, 512, 4, 1024, 0.1)
        self.svd = SVD()

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        # Extract features from both point clouds
        src, tgt = inputs
        src_feat = self.dgcnn(src)
        tgt_feat = self.dgcnn(tgt)

        # Produce new embeddings with attention model
        src_resid = self.transformer(tgt_feat, src_feat)
        tgt_resid = self.transformer(src_feat, tgt_feat)
        src_embed = src_feat + src_resid
        tgt_embed = tgt_feat + tgt_resid

        # Solve with SVD
        R, t = self.svd(src, tgt, src_embed, tgt_embed)
        T = tf.concat([R, t], 1)

        return T


class DCPLoss(keras.losses.Loss):
    def __init__(self, model: DCP, lamb: Optional[float] = None):
        super().__init__(reduction=keras.losses.Reduction.NONE)
        self.model = model
        self.lamb = lamb

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        # Compute transformation loss
        R_true, t_true = tf.split(y_true, [3, 1], axis=1)
        R_pred, t_pred = tf.split(y_pred, [3, 1], axis=1)
        batch_size = tf.cast(R_true.shape[0], tf.float32)
        R_diff = tf.matmul(R_pred, R_true, transpose_a=True) - tf.eye(3)
        R_err = tf.nn.l2_loss(R_diff)
        t_err = tf.nn.l2_loss(t_pred - t_true)
        loss = (R_err + t_err) / batch_size

        # Apply Tikhonov regularization
        if self.lamb is not None:
            for param in self.model.trainable_weights:
                loss += tf.nn.l2_loss(param) * self.lamb

        return loss


class DGCNN(keras.Model):
    def __init__(self, k: int = 16):
        super().__init__()

        self.graph = GraphFeature(k)

        from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization
        self.conv1 = Conv2D(64, 1, use_bias=False)
        self.conv2 = Conv2D(64, 1, use_bias=False)
        self.conv3 = Conv2D(128, 1, use_bias=False)
        self.conv4 = Conv2D(256, 1, use_bias=False)
        self.conv5 = Conv1D(512, 1, use_bias=False)

        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        self.bn4 = BatchNormalization()
        self.bn5 = BatchNormalization()

    def call(self, x: tf.Tensor):
        # Extract features from different levels
        x1 = tf.nn.relu(self.bn1(self.conv1(self.graph(x))))
        x1 = tf.reduce_max(x1, axis=-2)

        x2 = tf.nn.relu(self.bn2(self.conv2(self.graph(x1))))
        x2 = tf.reduce_max(x2, axis=-2)

        x3 = tf.nn.relu(self.bn3(self.conv3(self.graph(x2))))
        x3 = tf.reduce_max(x3, axis=-2)

        x4 = tf.nn.relu(self.bn4(self.conv4(self.graph(x3))))
        x4 = tf.reduce_max(x4, axis=-2)

        # Produce final embedding features
        x5 = tf.concat([x1, x2, x3, x4], axis=-1)
        x5 = tf.nn.relu(self.bn5(self.conv5(x5)))

        return x5  # (batch_size, num_vertices, num_features)


class GraphFeature(keras.layers.Layer):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def call(self, x: tf.Tensor):
        # Find k nearest neighbors for each point
        x_trans = tf.transpose(x, perm=(0, 2, 1))
        x_sq = tf.reduce_sum(x ** 2, axis=2, keepdims=True)
        pair_dist = 2 * tf.matmul(x, x_trans) - x_sq - \
            tf.transpose(x_sq, perm=(0, 2, 1))
        _, knn_idx = tf.nn.top_k(pair_dist, k=self.k)

        # Create edges to form a graph
        edge_1 = tf.repeat(tf.expand_dims(x, 2), self.k, axis=2)
        edge_2 = tf.gather(x, knn_idx, axis=1, batch_dims=1)
        edge = tf.concat([edge_1, edge_2], axis=-1)
        return edge  # (batch_size, num_vertices, k, 6)


class Transformer(keras.Model):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int,
                 rate: float):
        super().__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, rate)

    def call(self, src: tf.Tensor, tgt: tf.Tensor):
        enc_out = self.encoder(src)
        dec_out = self.decoder(tgt, enc_out)
        return dec_out


class Encoder(keras.layers.Layer):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int,
                 rate: float):
        super().__init__()
        self.layers = [EncoderLayer(d_model, num_heads, d_ff, rate)
                       for _ in range(num_layers)]

    def call(self, x: tf.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


class EncoderLayer(keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rate: float):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

        from tensorflow.keras.layers import LayerNormalization, Dropout
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x: tf.Tensor):
        # Multi-head attention
        atten_out = self.mha(x, x, x)
        atten_out = self.dropout1(atten_out)
        out1 = self.norm1(x + atten_out)

        # Feed forward
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        out2 = self.norm2(out1 + ffn_out)

        return out2


class Decoder(keras.layers.Layer):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int,
                 rate: float):
        super().__init__()
        self.layers = [DecoderLayer(d_model, num_heads, d_ff, rate)
                       for _ in range(num_layers)]

    def call(self, x: tf.Tensor, enc_out: tf.Tensor):
        for layer in self.layers:
            x = layer(x, enc_out)
        return x


class DecoderLayer(keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rate: float):
        super().__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

        from tensorflow.keras.layers import LayerNormalization, Dropout
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.norm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, x: tf.Tensor, enc_out: tf.Tensor):
        # Multi-head attention
        atten1 = self.mha1(x, x, x)
        atten1 = self.dropout1(atten1)
        out1 = self.norm1(atten1 + x)

        # Multi-head attention with encoder output as input
        atten2 = self.mha2(out1, enc_out, enc_out)
        atten2 = self.dropout2(atten2)
        out2 = self.norm2(atten2 + out1)

        # Feed forward
        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out)
        out3 = self.norm3(ffn_out + out2)

        return out3


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        from tensorflow.keras.layers import Dense
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)

    def call(self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor):
        # Linear layers for query, key and value
        batch_size = q.shape[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        # Split linear layers to multi-head
        # (batch_size, num_heads, seq_len, depth)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Apply scaled dot product attention
        scaled_atten = self.attention(q, k, v)
        scaled_atten = tf.transpose(scaled_atten, perm=(0, 2, 1, 3))

        # Concatenate head and final linear layer
        concat_atten = tf.reshape(scaled_atten, (batch_size, -1, self.d_model))
        output = self.dense(concat_atten)  # (batch_size, seq_len_q, d_model)
        return output

    def split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Split last dimension of `x` to (num_heads, depth) and transpose to
        # (batch_size, num_heads, seq_len, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=(0, 2, 1, 3))

    @staticmethod
    def attention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor,
                  mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        # Compute QK^T / sqrt(d_k)
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(k.shape[-1], tf.float32)
        scaled_atten_logits = matmul_qk / tf.math.sqrt(dk)

        # Apply mask
        if mask is not None:
            scaled_atten_logits += (mask * -1e9)

        # Normalize the last axis and multiply by V
        atten_wts = tf.nn.softmax(scaled_atten_logits, axis=-1)
        output = tf.matmul(atten_wts, v)

        return output


class FeedForward(keras.Sequential):
    def __init__(self, d_model: int, d_ff: int):
        from tensorflow.keras.layers import Dense
        super().__init__(layers=[
            Dense(d_ff, activation='relu'),
            Dense(d_model)
        ])


class SVD(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.identity = tf.eye(3)
        self.reflect = tf.constant([
            [1, 0, 0], [0, 1, 0], [0, 0, -1]
        ], dtype=tf.float32)

    def call(self, src: tf.Tensor, tgt: tf.Tensor, src_embed: tf.Tensor,
             tgt_embed: tf.Tensor):
        # Generate pointer
        batch_size, _, d_k = src.shape
        d_k = tf.cast(d_k, tf.float32)
        pointer = tf.matmul(tgt_embed, src_embed, transpose_b=True) / d_k
        pointer = tf.nn.softmax(pointer, axis=1)

        # Compute cross-variance matrix
        matched = tf.matmul(pointer, tgt, transpose_a=True)
        src_mean = tf.reduce_mean(src, axis=1, keepdims=True)
        src_demean = src - src_mean
        matched_mean = tf.reduce_mean(matched, axis=1, keepdims=True)
        matched_demean = matched - matched_mean
        H = tf.matmul(src_demean, matched_demean, transpose_a=True)

        # Use SVD to solve rotation
        _, U, V = tf.linalg.svd(H, full_matrices=True)
        R = tf.matmul(V, U, transpose_b=True)
        R_det = tf.linalg.det(R)
        R_pos = tf.greater_equal(R_det, 0)
        R_pos = tf.expand_dims(tf.expand_dims(R_pos, -1), -1)
        R_pos = tf.tile(R_pos, [1, 3, 3])
        refl = tf.where(R_pos, self.identity, self.reflect)
        R = tf.matmul(refl, R)

        # Compute translation
        t = tf.matmul(src_mean, -R, transpose_b=True) + matched_mean

        return (R, t)


if __name__ == "__main__":
    batch_size = 2
    import numpy as np
    model = DCP()
    x = np.random.randn(batch_size, 2048, 3).astype(np.float32)
    R = np.expand_dims(np.eye(3), 0)
    R = np.tile(R, (batch_size, 1, 1))
    t = np.zeros((batch_size, 1, 3))
    T = np.concatenate([R, t], axis=1)
    model.compile(optimizer='adam', loss=DCPLoss(model))
    print(model.train_on_batch(x=(x, x), y=T))
