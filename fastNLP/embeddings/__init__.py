r"""
embeddings 模块主要用于从各种预训练的模型中获取词语的分布式表示，目前支持的预训练模型包括word2vec, glove, ELMO, BERT等。这里所有
embedding的forward输入都是形状为 ``(batch_size, max_len)`` 的torch.LongTensor，输出都是 ``(batch_size, max_len, embedding_dim)`` 的
torch.FloatTensor。所有的embedding都可以使用 `self.num_embedding` 获取最大的输入index范围, 用 `self.embeddig_dim` 或 `self.embed_size` 获取embedding的
输出维度。
"""

__all__ = [
    "Embedding",
    "TokenEmbedding",
    "StaticEmbedding",
    "ElmoEmbedding",
    "BertEmbedding",
    "BertWordPieceEncoder",

    "RobertaEmbedding",
    "RobertaWordPieceEncoder",

    "GPT2Embedding",
    "GPT2WordPieceEncoder",

    "StackEmbedding",
    "LSTMCharEmbedding",
    "CNNCharEmbedding",

    "get_embeddings",
    "get_sinusoid_encoding_table"
]

import sys

from .bert_embedding import BertEmbedding
from .bert_embedding import BertWordPieceEncoder
from .char_embedding import CNNCharEmbedding
from .char_embedding import LSTMCharEmbedding
from .elmo_embedding import ElmoEmbedding
from .embedding import Embedding
from .embedding import TokenEmbedding
from .gpt2_embedding import GPT2Embedding
from .gpt2_embedding import GPT2WordPieceEncoder
from .roberta_embedding import RobertaEmbedding
from .roberta_embedding import RobertaWordPieceEncoder
from .stack_embedding import StackEmbedding
from .static_embedding import StaticEmbedding
from .utils import get_embeddings
from .utils import get_sinusoid_encoding_table
from ..doc_utils import doc_process

doc_process(sys.modules[__name__])
