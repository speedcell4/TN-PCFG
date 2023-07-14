r"""
.. todo::
    doc
"""

__all__ = [
    "ConvolutionCharEncoder",
    "LSTMCharEncoder",

    "ConvMaxpool",

    "LSTM",

    "StarTransformer",

    "TransformerEncoder",

    "VarRNN",
    "VarLSTM",
    "VarGRU",

    "MaxPool",
    "MaxPoolWithMask",
    "KMaxPool",
    "AvgPool",
    "AvgPoolWithMask",

    "MultiHeadAttention",
    "BiAttention",
    "SelfAttention",

    "BertModel",

    "RobertaModel",

    "GPT2Model",

    "LSTMSeq2SeqEncoder",
    "TransformerSeq2SeqEncoder",
    "Seq2SeqEncoder"
]

from fastNLP.modules.attention import BiAttention
from fastNLP.modules.attention import MultiHeadAttention
from fastNLP.modules.attention import SelfAttention
from .bert import BertModel
from .char_encoder import ConvolutionCharEncoder
from .char_encoder import LSTMCharEncoder
from .conv_maxpool import ConvMaxpool
from .gpt2 import GPT2Model
from .lstm import LSTM
from .pooling import AvgPool
from .pooling import AvgPoolWithMask
from .pooling import KMaxPool
from .pooling import MaxPool
from .pooling import MaxPoolWithMask
from .roberta import RobertaModel
from .seq2seq_encoder import LSTMSeq2SeqEncoder
from .seq2seq_encoder import Seq2SeqEncoder
from .seq2seq_encoder import TransformerSeq2SeqEncoder
from .star_transformer import StarTransformer
from .transformer import TransformerEncoder
from .variational_rnn import VarGRU
from .variational_rnn import VarLSTM
from .variational_rnn import VarRNN
