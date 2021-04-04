import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout

from attention import  Attention
from add_normalization import AddNormalizationWrapper, LayerNormalization
from feed_forward_network import FeedForwardNetwork

class Decoder(keras.models.Model):
    """
    Decoder

    """

    def __init__(self, vocab_size: int, hidden_dim: int, dropout_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self_attention_base_layer = Attention(depth=hidden_dim)
        src_tar_attention_base_layer = Attention(depth=hidden_dim)
        ffn_base_layer = FeedForwardNetwork(hidden_dim=hidden_dim, dropout_rate=dropout_rate)

        self.self_attention = AddNormalizationWrapper(self_attention_base_layer, dropout_rate)
        self.src_tar_attention = AddNormalizationWrapper(src_tar_attention_base_layer, dropout_rate)
        self.ffn = AddNormalizationWrapper(ffn_base_layer, dropout_rate)
        self.output_normalization = LayerNormalization()
        self.output_dropout = Dropout(dropout_rate)

    def call(self, input: tf.Tensor, enc_output: tf.Tensor):
        # Self-Attention層
        self_attention_output = self.self_attention(input=input, memory=input)
        # Source-Target Attention層
        src_tar_attention_output = self.src_tar_attention(input=self_attention_output, memory=enc_output)
        # Position-wise Feed Forwardネットワーク層
        ffn_output = self.ffn(input=src_tar_attention_output)
        # レイヤノーマライゼーション&ドロップアウト
        output = self.output_normalization(ffn_output)
        return self.output_dropout(output)


if __name__ == '__main__':
    from encoder import Encoder
    from word_embedding import WordEmbedding as we
    vocab_size = 10
    emb_dim = 2
    dor = 0.1

    # Encoder側
    einputs = tf.constant([[0, 2, 3], [1, 4, 3]])
    enc_emb = we(vocab_size=vocab_size, embedding_dim=emb_dim)
    enc_inputs = enc_emb(einputs)
    encoder = Encoder(vocab_size=10, hidden_dim=2, dropout_rate=dor)
    enc_output = encoder(enc_inputs)

    # Dec側
    dinputs = tf.constant([[2, 0, 3], [4, 1, 3]])
    dec_emb = we(vocab_size=vocab_size, embedding_dim=emb_dim)
    dec_inputs = dec_emb(dinputs)
    decoder = Decoder(vocab_size=10, hidden_dim=2, dropout_rate=dor)
    dec_output = decoder(dec_inputs, enc_output)
    print(dec_output)
