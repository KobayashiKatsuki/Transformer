import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding

from positional_encoding import PositionalEncoding

# ゼロパディングのID
PAD_ID = 3


class WordEmbedding(keras.models.Model):
    """
    単語埋め込みクラス
    入力テンソルの埋め込み＋Positional Encodingを行う
    """
    def __init__(self, vocab_size: int, embedding_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_embedding = TokenEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim)
        self.positional_encoding = PositionalEncoding()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        emb_input = self.token_embedding(inputs)
        pos_enc_input = self.positional_encoding(emb_input)
        return pos_enc_input


class TokenEmbedding(keras.layers.Layer):
    """
    埋め込み層
    [バッチサイズ, 系列長（単語数）] というTensorで受け取ると
    [バッチサイズ, 系列長, 埋め込み次元数] というTensorで出力する

    tf.nn.embedding_lookup(table, indices) はただidicesで指定の埋め込みベクトルをtableから引くだけの関数
    keras.layers.Embedding は tf.nn.embedding_lookupをラッピングしているのでこっち使ったほうが良いような
    
    """
    def __init__(self, vocab_size: int, embedding_dim: int, dtype=tf.float32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dtype_ = dtype

        self.embedding_layer = Embedding(
            input_dim = vocab_size,
            output_dim = embedding_dim,
            name='embedding'
        )


    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        embedding = self.embedding_layer(inputs)
        #embedding *= tf.expand_dims(mask, -1)  # 元々 PAD だった部分を0にする
        #embeding = embedding * self.embedding_dim ** 0.5  # scaling
        return embedding


if __name__ == '__main__':
    # 2次元空間への埋め込み層
    emb = WordEmbedding(vocab_size = 10, embedding_dim=2)

    # 0, 2, 3 というトークン列と 1, 4, 3 というトークン列
    # batch_sizeは2, max_lenは3
    x_tensor = tf.constant([[0, 2, 3], [1, 4, 3]])
    emb_tensor = emb(x_tensor)
    print(emb_tensor)
