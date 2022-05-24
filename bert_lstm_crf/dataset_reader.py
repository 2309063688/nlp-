from typing import Dict, Iterable, List
import logging

from overrides import overrides

from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field
from allennlp.data import (
    DatasetReader,
    Instance,
)
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import TokenIndexer, PretrainedTransformerMismatchedIndexer
from allennlp.data.tokenizers import Token ,Tokenizer


# 设置日志对象
logger = logging.getLogger(__name__)
# 选择transformer模型
transformers_model = "bert-base-chinese"


class SequenceTaggingDatasetReader(DatasetReader):
    def __init__(
        self,
        tokenizers: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        **kwargs,
    ) -> None:
        '''
        初始化
        :param tokenizers:分词方式
        :param token_indexers:id化方式
        :param kwargs:
        '''
        super().__init__(**kwargs)
        self._tokenizers = tokenizers or PretrainedTransformerTokenizer(model_name=transformers_model)
        self._token_indexers = token_indexers or {"tokens": PretrainedTransformerMismatchedIndexer(model_name=transformers_model)}

    @overrides
    def _read(self, file_path) -> Iterable[Instance]:
        '''
        读取数据
        :param file_path: 数据文件路径
        :return:
        Instance组成的可遍历对象
        '''
        start = 0
        with open(file_path, "r", encoding="utf-8") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            lines = [line for line in data_file]
            for i, line in enumerate(lines):
                if len(lines[i]) <= 1:
                    tokens_and_tags = [l.split()[1:4:2] for l in lines[start:i]]
                    start = i + 1
                    while start < len(lines) and len(lines[start]) <= 1:
                        start += 1
                    text = [token for token, tag in tokens_and_tags]
                    tokens = [Token(token) for token in text]
                    tags = [tag for token, tag in tokens_and_tags]
                    yield self.text_to_instance(tokens, tags)

    def text_to_instance(
            self, tokens: List[Token], tags: List[str] = None
    ) -> Instance:
        '''
        将文本转为instance
        :param tokens: token组成的列表
        :param tags: 各token对应的词性组成的列表
        :return:
        由文本内容转化而来的Instance对象
        '''
        fields: Dict[str, Field] = {}
        sequence = TextField(tokens, self._token_indexers)
        fields["tokens"] = sequence
        fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})
        if tags is not None:
            fields["tags"] = SequenceLabelField(tags, sequence)
        return Instance(fields)

# 测试
# reader = SequenceTaggingDatasetReader()
# training_data = list(reader.read("data/train.conll"))
