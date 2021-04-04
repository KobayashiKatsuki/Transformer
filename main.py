import pandas as pd
import sentencepiece as spm
import tensorflow as tf
from tensorflow import  keras
from transformer import Transformer

#%%

def preprocess():
    """
    sentencepieceやってみる
    sentencepieceはサブワードを含む単語分割を行うモジュール

    本タスクの事前にどっか別のところでやっておく
    """
    df_all = pd.read_csv('D:/DataSet/jpn-eng/jpn_mini.txt', sep='\t', header=None)
    df_all[0].to_csv('D:/DataSet/jpn-eng/eng_sentences.txt', header=None, index=False)
    df_all[1].to_csv('D:/DataSet/jpn-eng/jpn_sentences.txt', header=None, index=False)

    # モデル作成
    # ・--input : テキストデータ（1文が1行）のファイルパス
    # ・--model_prefix : 出力ファイル名
    # ・--vocab_size : 語彙数。小規模は数千から10k、超大規模は32k程度
    # ・--character_coverage : モデルがカバーする語彙割合。1.0で100%カバー。日本語は0.9995推奨。
    # ・--model_type : モデル種別 (unigram(デフォルト), bpe)
    # ・--pad_id : <pad>に割り当てるID

    # 日本語
    spm.SentencePieceTrainer.Train(
        "--input=D:/DataSet/jpn-eng/jpn_sentences.txt --model_prefix=japanese --character_coverage=0.9995 --vocab_size=250 --pad_id=3"
    )

    # 英語
    spm.SentencePieceTrainer.Train(
        "--input=D:/DataSet/jpn-eng/eng_sentences.txt --model_prefix=english --vocab_size=150 --pad_id=3"
    )

    # サブワード使ってみる
    sp = spm.SentencePieceProcessor()
    sp.Load('japanese.model')

    input_text = '君はコーヒーが好き？'
    print(sp.EncodeAsPieces(input_text))
    print(sp.EncodeAsIds(input_text))

#%%
def create_encoder_decoder_data(max_len=20):
    """
    Encoder, Decoderの作成

    データセットと逆だが
    日本語（Enc）-> 英語（Dec）という翻訳タスクとする
    """
    df_all = pd.read_csv('D:/DataSet/jpn-eng/jpn_mini.txt', sep='\t', header=None)

    def sentence_to_tokens(model: str, df_sentence: pd.DataFrame, max_len=20, is_enc=True):
        """
        sentenceをToken列にする
        encoderなら(is_enc==True)そのまま、
        decoderなら(is_enc==False)前後にBOS, EOSを付加する

        """
        sp = spm.SentencePieceProcessor()
        sp.Load(model)

        bos = sp.PieceToId('<s>')
        eos = sp.PieceToId('</s>')
        pad = sp.PieceToId('<pad>')

        for idx in range(df_sentence.shape[0]):
            sentence = df_sentence.values[idx]
            tokens = sp.EncodeAsIds(sentence)
            if is_enc is True:
                tokens = tokens[:max_len]
            else:
                tokens = [bos] + tokens[:max_len-2] + [eos]

            tokens = tokens + [pad] * (max_len - len(tokens))
            yield tokens

    def tokens_to_sentence(model: str, tokens: list ):
        """
        token列からsentenceを復元
        """
        sp = spm.SentencePieceProcessor()
        sp.Load(model)
        sentence = ''
        for t in tokens:
            s = sp.IdToPiece(t)
            sentence = sentence + s
        return sentence

    # 日本語（Encoder側）
    jpn_tokens = [jt for jt in sentence_to_tokens(model='japanese.model', df_sentence=df_all[1], max_len=10)]
    # 英語（Decoder側）
    eng_tokens = [et for et in sentence_to_tokens(model='english.model', df_sentence=df_all[0], max_len=10, is_enc=False)]

    # for jt in jpn_tokens:
    #     js = tokens_to_sentence(model='japanese.model', tokens=jt)
    #     print(js)
    # for et in eng_tokens:
    #     es = tokens_to_sentence(model='english.model', tokens=et)
    #     print(es)

    return tf.constant(jpn_tokens[:5]), tf.constant(eng_tokens[:5])


#%%
if __name__ == '__main__':
    # 前処理
    #preprocess()
    
    # 学習データの生成
    enc_input, dec_input = create_encoder_decoder_data()
    
    #print(enc_input)
    #print(dec_input)

    
    # Transformerモデルの構築
    #transformer = Transformer()










# %%
