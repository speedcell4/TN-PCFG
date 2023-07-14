r"""
fastNLP 在 :mod:`~fastNLP.models` 模块中内置了如 :class:`~fastNLP.models.CNNText` 、
:class:`~fastNLP.models.SeqLabeling` 等完整的模型，以供用户直接使用。

.. todo::
    这些模型的介绍（与主页一致）


"""
__all__ = [
    "CNNText",

    "SeqLabeling",
    "AdvSeqLabel",
    "BiLSTMCRF",

    "ESIM",

    "StarTransEnc",
    "STSeqLabel",
    "STNLICls",
    "STSeqCls",

    "BiaffineParser",
    "GraphParser",

    "BertForSequenceClassification",
    "BertForSentenceMatching",
    "BertForMultipleChoice",
    "BertForTokenClassification",
    "BertForQuestionAnswering",

    "TransformerSeq2SeqModel",
    "LSTMSeq2SeqModel",
    "Seq2SeqModel",

    'SequenceGeneratorModel'
]

import sys

from .base_model import BaseModel
from .bert import BertForMultipleChoice
from .bert import BertForQuestionAnswering
from .bert import BertForSentenceMatching
from .bert import BertForSequenceClassification
from .bert import BertForTokenClassification
from .biaffine_parser import BiaffineParser
from .biaffine_parser import GraphParser
from .cnn_text_classification import CNNText
from .seq2seq_generator import SequenceGeneratorModel
from .seq2seq_model import LSTMSeq2SeqModel
from .seq2seq_model import Seq2SeqModel
from .seq2seq_model import TransformerSeq2SeqModel
from .sequence_labeling import AdvSeqLabel
from .sequence_labeling import BiLSTMCRF
from .sequence_labeling import SeqLabeling
from .snli import ESIM
from .star_transformer import StarTransEnc
from .star_transformer import STNLICls
from .star_transformer import STSeqCls
from .star_transformer import STSeqLabel
from ..doc_utils import doc_process

doc_process(sys.modules[__name__])
