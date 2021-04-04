import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout

class FeedForwardNetwork(tf.keras.models.Model):
    '''
    Position-wise Feedforward Neural Network
    '''
    def __init__(self, hidden_dim: int, dropout_rate: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.filter_dense_layer = Dense(hidden_dim * 4, use_bias=True, activation=tf.nn.relu, name='filter_layer')
        self.output_dense_layer = Dense(hidden_dim, use_bias=True, name='output_layer')
        self.dropout_layer = Dropout(dropout_rate) # Dropout層の実装

    def call(self, input: tf.Tensor, training=True) -> tf.Tensor:
        '''
        FeedForwardNetwork を適用します。
        :param input: shape = [batch_size, length, hidden_dim]
        :return: shape = [batch_size, length, hidden_dim]
        '''
        tensor = self.filter_dense_layer(input)
        tensor = self.dropout_layer(tensor, training=training)
        return self.output_dense_layer(tensor)


if __name__ == '__main__':
    ffn = FeedForwardNetwork(hidden_dim=2, dropout_rate=0.1)
    x = tf.constant([
        [[1, 2], [3, 4], [5, 6]], 
        [[3, 4], [5, 6], [1, 2]]
    ])
    print(x)
    y = ffn(x)
    print(y)

