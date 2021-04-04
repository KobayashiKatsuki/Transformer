import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Activation

from attention import  Attention
from add_normalization import AddNormalizationWrapper, LayerNormalization
from feed_forward_network import FeedForwardNetwork

class Encoder(keras.models.Model):
    """
    Encoder

    入力（埋め込み後）Xに対するSelfAttentionを行う
    """

    def __init__(self, vocab_size: int, hidden_dim: int, dropout_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        attention_base_layer = Attention(depth=hidden_dim)
        ffn_base_layer = FeedForwardNetwork(hidden_dim=hidden_dim, dropout_rate=dropout_rate)

        self.attention = AddNormalizationWrapper(attention_base_layer, dropout_rate)
        self.ffn = AddNormalizationWrapper(ffn_base_layer, dropout_rate)
        self.output_normalization = LayerNormalization()

    def call(self, input: tf.Tensor):
        # Self-Attention層
        # 出力はinputに対してattentionの重みが加算されたもの
        attention_output = self.attention(input=input, memory=input)

        # Position-wise Feed Forwardネットワーク層
        # 単語列の位置毎に独立処理する全結合FFN
        ffn_output = self.ffn(input=attention_output)

        # 最終層にレイヤノーマライゼーションをかけて出力
        return self.output_normalization(ffn_output)



if __name__ == '__main__':

    vocab_size = 10
    emb_dim = 2
    dor = 0.1

    # トークン列
    inputs = tf.constant([[0, 2, 3], [1, 4, 3]])

    # embeddingまでの処理
    import word_embedding
    emb = word_embedding.WordEmbedding(vocab_size=vocab_size, embedding_dim=emb_dim)
    enc_inputs = emb(inputs)
    #print(enc_inputs)
    
    # Encoder
    encoder = Encoder(vocab_size=10, hidden_dim=2, dropout_rate=dor)
    enc_output = encoder(enc_inputs)
    print(enc_output)

