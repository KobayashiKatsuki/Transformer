import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Activation

class Attention(keras.models.Model):
    """
    Attentionクラス
    """

    def __init__(self, depth: int,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = depth # トークンの次元数（単語なら埋め込み次元）

        # 全結合層: Dense(出力側ユニット数(#入力側は積層する層に依存), 他引数)
        # ここでは入出力ともdepth次元（q,k,v_dense.shape = [depth, depth]）
        self.q_dense_layer = Dense(depth, use_bias=False, name='q_dense_layer')
        self.k_dense_layer = Dense(depth, use_bias=False, name='k_dense_layer')
        self.v_dense_layer = Dense(depth, use_bias=False, name='v_dense_layer')
        self.output_dense_layer = Dense(depth, use_bias=False, name='output_dense_layer')


    def call(self, input: tf.Tensor, memory: tf.Tensor) -> tf.Tensor:
            '''
            モデルの実行を行います。
            :param input: query のテンソル
            :param memory: query に情報を与える memory のテンソル
            :return: inputのどの単語がどのくらいの注意を持つかを表す重み（attention）

            call関数は特殊な関数。レイヤを関数風に`コール`したときに呼ばれる。
            attention_output = attention(input=x, memory=y)
            みたいな

            他にもcall実行時に一度だけ呼ばれるbuild関数てのがある
            '''
            q = self.q_dense_layer(input)  # [batch_size, q_length, depth]
            k = self.k_dense_layer(memory)  # [batch_size, m_length, depth]
            v = self.v_dense_layer(memory)

            # ここで q と k の内積を取ることで、query と key の関連度のようなものを計算します。
            q *= self.depth ** -0.5  # scaled dot-product
            logit = tf.matmul(q, k, transpose_b=True)  # [batch_size, q_length, k_length]

            # softmax を取ることで正規化します
            attention_weight = tf.nn.softmax(logit, name='attention_weight')
            """
            attention_weightは 各queryに対する各keyへのattentionの行列
            [    k1  k2  km    
            q1 [a11 a12 a1m]
            q2 [a21 a22 a2m]
            qn [an1 an2 anm]
            ]
            """

            # 重みに従って value から情報を引いてきます
            attention_output = tf.matmul(attention_weight, v)  # [batch_size, q_length, depth]
            """
            valueは各keyとペアになるベクトル(d次元)
            [
            v1 [v11 v12 v1d]
            v2 [v21 v22 v2d]
            vm [vm1 vm2 vmd]
            ]
            ここであるqueryに対するattentionを作用させると
            attention_weight[qn] = [an1 an2 anm]
            なので
            matmul(attention_weight, v)
            = an1*v1 + an2*v2 + anm*vm
            = an1*[v11 v12 v1d] + an2*[v21 v22 v2d] + anm*[vm1 vm2 vmd]
            各vの重み付き和のベクトルになる(コンテキストベクトルと呼ばれるもの)
            """
            return self.output_dense_layer(attention_output)


if __name__ == '__main__':
    attention_layer = Attention(depth=1)

    X = tf.constant([
        [[0, 1], [2, 3], [4, 5],],
        [[4, 5], [6, 7], [4, 5],],
        [[8, 9], [10, 11], [4, 5],],
        [[12, 13], [14, 15], [4, 5],],
    ])

    #attention_output = attention_layer(input=x, memory=y) # Source-Target Attention
    attention_output = attention_layer(input=X, memory=X) # Self Attention
    
    print(attention_output)
