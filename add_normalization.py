import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout

class AddNormalizationWrapper(tf.keras.models.Model):
    '''
    Add & Normalization

    残差結合してレイヤ正規化をかけるラッパークラス

    '''
    def __init__(self, base_layer: tf.keras.layers.Layer, dropout_rate: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer = base_layer
        self.layer_normalization = LayerNormalization()
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input: tf.Tensor, training: bool, *args, **kwargs) -> tf.Tensor:
        tensor = self.layer_normalization(input)
        tensor = self.layer(tensor, training=training, *args, **kwargs)
        tensor = self.dropout_layer(tensor, training=training)
        return tf.add(input, tensor)


class LayerNormalization(tf.keras.layers.Layer):
    '''
    レイヤーノーマライゼーションです。
    レイヤーの出力が平均 bias, 標準偏差 scale になるように調整します。
    レイヤはテンソルの中の一番深い層（1トークンのベクトル）に相当
    '''
    def build(self, input_shape: tf.TensorShape) -> None:
        hidden_dim = input_shape[-1]
        self.scale = self.add_weight('layer_norm_scale', shape=[hidden_dim],
                                    initializer=tf.ones_initializer())
        self.bias = self.add_weight('layer_norm_bias', [hidden_dim],
                                    initializer=tf.zeros_initializer())
        super().build(input_shape)

    def call(self, x: tf.Tensor, epsilon: float = 1e-6) -> tf.Tensor:
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)

        return norm_x * self.scale + self.bias


if __name__ == '__main__':
    from feed_forward_network import FeedForwardNetwork

    do_rate = 0.1

    ffn = FeedForwardNetwork(hidden_dim=2, dropout_rate=do_rate)
    ffn_addnorm = AddNormalizationWrapper(ffn, do_rate)

    x = tf.constant([
        [[1., 2.], [3., 4.]], 
        [[1., 3.], [2., 4.]]
    ])
    print(x)
    y = ffn_addnorm(x)
    print(y)

