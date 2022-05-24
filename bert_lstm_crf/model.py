from allennlp.data.fields import MetadataField

from dataset_reader import SequenceTaggingDatasetReader

import tempfile
from typing import Dict, Iterable, List, Tuple, Optional, Any

import numpy
import torch
import torch.nn.functional
from TorchCRF import CRF
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.token_embedders import Embedding, PretrainedTransformerMismatchedEmbedder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, LstmSeq2VecEncoder
from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder
from allennlp.nn import util, InitializerApplicator
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.training import Trainer, GradientDescentTrainer
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.util import evaluate


# 选择transformer模型
transformer_model = "bert-base-chinese"


class SimpleSequenceTagging(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            embedder: PretrainedTransformerMismatchedEmbedder,
            encoder: LstmSeq2SeqEncoder,
            calculate_span_f1: bool = None,
            label_encoding: Optional[str] = None,
            label_namespace: str = "labels",
            verbose_metrics: bool = True,
            initializer: InitializerApplicator = InitializerApplicator(),
            **kwargs,
    ):
        '''
        模型初始化
        :param vocab:词典
        :param embedder:将token转为词嵌入的方式
        :param encoder: 将词嵌入转为词向量的方式
        :param calculate_span_f1: 是否用f1分数进行评价
        :param label_encoding: 标签的向量？
        :param label_namespace:词典中 标签所在空间名
        :param verbose_metrics:是否输出正确率日志信息
        :param initializer: 用特定方式初始化模型参数
        :param kwargs:
        '''
        super().__init__(vocab, **kwargs)

        self.embedder = embedder
        self.encoder = encoder
        self.label_namespace = label_namespace
        self.num_classes = vocab.get_vocab_size(label_namespace)
        self._verbose_metrics = verbose_metrics

        # 分类器
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), self.num_classes)
        # crf层
        self.crf = CRF(self.num_classes, batch_first=True)

        # 检测embedder输出和encoder输入的张量维度是否相等
        check_dimensions_match(
            embedder.get_output_dim(),
            encoder.get_input_dim(),
            "text field embedding dim",
            "encoder input dim",
        )

        # 用acc和acc3评价模型
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3),
        }

        if calculate_span_f1 is None:
            calculate_span_f1 = label_encoding is not None

        self.calculate_span_f1 = calculate_span_f1
        self._f1_metric: Optional[SpanBasedF1Measure] = None
        if calculate_span_f1:
            if not label_encoding:
                raise ConfigurationError(
                    "calculate_span_f1 is True, but no label_encoding was specified."
                )
            self._f1_metric = SpanBasedF1Measure(
                vocab, tag_namespace=label_namespace, label_encoding=label_encoding
            )

        initializer(self)

    def forward(
            self,
            tokens: TextFieldTensors,
            tags: torch.LongTensor = None,
            metadata: List[Dict[str, Any]] = None,
            ignore_loss_on_o_tags: bool = False,
    ) -> Dict[str, torch.Tensor]:
        '''
        前向算法
        :param tokens: 数据域，形式为{"tokens":
                                        {"tokens":
                                            ...
                                        }
                                    }
        :param tags: 标签张量
        :param metadata: 未id化的文本内容
        :param ignore_loss_on_o_tags:
        :return:
        返回一个out字典，包含logits，probs，loss和words
        '''
        # shape: (batch_size, num_tokens, embedding_dim) 转为词嵌入
        a = tokens['tokens']
        embedded_text = self.embedder(**a)
        batch_size, sequence_length, _ = embedded_text.size()
        # Shape: (batch_size, num_tokens) 获取mask
        mask = util.get_text_field_mask(tokens)
        # Shape: (batch_size, encoding_dim) 转为词向量
        encoded_text = self.encoder(embedded_text, mask)

        # Shape: (batch_size, num_labels) 得分
        logits = self.classifier(encoded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        # Shape: (batch_size, num_labels) softmax将得分转为概率
        probs = torch.nn.functional.softmax(reshaped_log_probs, dim=-1).view(
            [batch_size, sequence_length, self.num_classes]
        )

        output = {"logits": logits, "probs": probs}

        # 若存在labels，计算loss
        if tags is not None:
            if ignore_loss_on_o_tags:
                o_tag_index = self.vocab.get_token_index("O", namespace=self.label_namespace)
                tag_mask = mask & (tags != o_tag_index)
            else:
                tag_mask = mask
            output["loss"] = self.crf(logits, tags, tag_mask) * -1

            for metric in self.metrics.values():
                metric(logits, tags, mask)
            if self.calculate_span_f1:
                self._f1_metric(logits, tags, mask)

        # 存储原数据
        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]
        return output

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        '''
        使输出可读化
        :param output_dict: 输出字典
        :return:
        out字典，其中添加了预测的具体标签
        '''
        all_predictions = output_dict["probs"]
        all_predictions = all_predictions.cpu().data.numpy()
        if all_predictions.ndim == 3:
            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        for predictions in predictions_list:
            argmax_indices = numpy.argmax(predictions, axis=-1)
            tags = [
                self.vocab.get_token_from_index(x, namespace=self.label_namespace)
                for x in argmax_indices
            ]
            all_tags.append(tags)
        all_tags = numpy.array(all_tags)
        all_tags = torch.from_numpy(all_tags)
        output_dict["tags"] = all_tags
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        '''
        获取metrics
        :param reset: 是否在一个epoch结束后将loss，acc，f1清零
        :return:
        metircs
        '''
        metrics_to_return = {
            metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()
        }

        if self.calculate_span_f1:
            f1_dict = self._f1_metric.get_metric(reset)  # type: ignore
            if self._verbose_metrics:
                metrics_to_return.update(f1_dict)
            else:
                metrics_to_return.update({x: y for x, y in f1_dict.items() if "overall" in x})
        return metrics_to_return


def build_dataset_reader() -> DatasetReader:
    '''
    建立数据读取类
    :return: 数据读取类
    '''
    return SequenceTaggingDatasetReader()


def read_data(reader: DatasetReader) -> Tuple[List[Instance], List[Instance]]:
    '''
    从文件中中读取数据
    :param reader: 数据读取类
    :return:
    各数据集数据转化成的Instance列表
    '''
    print("Reading data")
    training_data = list(reader.read("data/train.conll"))
    validation_data = list(reader.read("data/dev.conll"))
    return training_data, validation_data


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    '''
    建立词典
    :param instances: 可遍历Instance对象
    :return:
    词典
    '''
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)


def build_model(vocab: Vocabulary) -> Model:
    '''
    建立模型
    :param vocab: 词典
    :return:
    模型
    '''
    print("Building the model")
    embedder = PretrainedTransformerMismatchedEmbedder(
        model_name=transformer_model
    )
    encoder = LstmSeq2SeqEncoder(
        input_size=768,
        hidden_size=768,
        num_layers=2,
        dropout=0.5,
        bidirectional=True
    )
    return SimpleSequenceTagging(vocab, embedder, encoder)


def run_training_loop():
    '''
    训练函数
    :return:
    训练好的模型和训练数据
    '''
    dataset_reader = build_dataset_reader()

    train_data, dev_data = read_data(dataset_reader)

    vocab = build_vocab(train_data + dev_data)
    model = build_model(vocab)

    train_loader, dev_loader = build_data_loader(train_data, dev_data)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    with tempfile.TemporaryDirectory() as serialization_dir:
        trainer = build_trainer(model, serialization_dir, train_loader, dev_loader)
        print("Start training")
        trainer.train()
        print("Finished training")

    return model, dataset_reader


def build_data_loader(
        train_data: List[Instance],
        dev_data: List[Instance],
) -> Tuple[DataLoader, DataLoader]:
    '''
    将数据按batchsize组合
    :param train_data: 训练集数据
    :param dev_data: 开发集数据
    :return:
    组合后的数据
    '''
    train_loader = SimpleDataLoader(train_data, 8, shuffle=True)
    dev_loader = SimpleDataLoader(dev_data, 8, shuffle=False)
    return train_loader, dev_loader


def build_trainer(
        model: Model,
        serialization_dir: str,
        train_loader: DataLoader,
        dev_loader: DataLoader,
) -> Trainer:
    '''
    建立训练器
    :param model: 模型
    :param serialization_dir: 指定包括checkpoint，vocab等存储地址
    :param train_loader: 训练集数据
    :param dev_loader: 开发集数据
    :return:
    训练器
    '''
    parameters = [(n,p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters, lr=0.0001)
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=100,
        optimizer=optimizer,
        cuda_device=-1,
    )
    return trainer


model, dataset_reader = run_training_loop()
vocab = model.vocab
test_data = list(dataset_reader.read("data/dev.conll"))
data_loader = SimpleDataLoader(test_data, batch_size=8)
data_loader.index_with(model.vocab)
results = evaluate(model, data_loader)