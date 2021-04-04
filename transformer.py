import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Activation
from tensorflow.keras.preprocessing.text import Tokenizer

from word_embedding import WordEmbedding
from encoder import Encoder
from decoder import Decoder


class Transformer(keras.models.Model):
    """
    Transformer

    構造
        encoder input(tokens)
                    ↓
        WordEmbedding(TokenEmbedding + Positional Encoding)
                    ↓
        Encoder(SelfAttention + Feed Forward Network)
                    ↓ 
    decoder input   ↓
        ↓           ↓
    WordEmbedding → Decoder(Masked Attention)
    """

    def __init__(
            self, 
            vocab_size: int, 
            head_num: int = 8,
            hidden_dim: int = 512, 
            dropout_rate: float = 0.1, 
            max_len: int = 50,
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        # Encoder側埋め込み層
        self.enc_embedding = WordEmbedding(vocab_size=vocab_size, embedding_dim=hidden_dim)
        # Encoder
        self.encoder = Encoder(vocab_size=vocab_size, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        
        # Decoder側埋め込み層
        self.dec_embedding = WordEmbedding(vocab_size=vocab_size, embedding_dim=hidden_dim)
        # Decoder
        self.decoder = Decoder(vocab_size=vocab_size, hidden_dim=hidden_dim, dropout_rate=dropout_rate)

    def call(self, encoder_input: tf.Tensor, decoder_input: tf.Tensor):
        # Encoder側
        encoder_input = self.enc_embedding(encoder_input)
        encoder_output = self.encoder(encoder_input)
        # Decoder側
        decoder_input = self.dec_embedding(decoder_input)
        decoder_output = self.decoder(decoder_input, encoder_output)
        print(decoder_output)


if __name__ == '__main__':
    vocab_size = 10
    hidden_dim = 2
    max_len = 5

    transformer = Transformer(
        vocab_size = vocab_size,
        hidden_dim = hidden_dim,
        max_len = max_len
    )

    enc_input = tf.constant([[0, 2, 3], [1, 4, 3]])
    dec_input = tf.constant([[2, 0, 3], [4, 1, 3]])

    transformer(enc_input, dec_input)

"""
これ何？

graph = tf.Graph()
with graph.as_default():
    transformer = Transformer(
        vocab_size=vocab_size,
        hopping_num=4,
        head_num=8,
        hidden_dim=512,
        dropout_rate=0.1,
        max_length=50,
    )
    transformer.build_graph()

"""