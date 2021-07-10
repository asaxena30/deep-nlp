from collections import Counter
from typing import List, Tuple

import nltk
import torch
from pycocotools.coco import COCO
from torch import nn as nn
from torch.nn import AdaptiveAvgPool2d
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter

from src.common.neural_net_param_utils import init_lstm_forget_gate_biases
from src.main.common.vocabulary import Vocab
from src.main.mscoco.coco_dataset import CustomCocoDataset
from src.modules.attention.attention_modules import BidirectionalAttention
from src.modules.common.resnet_models import ResNetModelType, get_resnet_model, ResNetWithLastLayerModified


def get_loader(data_path: str, coco_json_path: str, vocabulary: Vocab, transform, batch_size: int, shuffle: bool,
               num_workers: int, clean_annotations_data: bool = False):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco_dataset = CustomCocoDataset(data_path=data_path,
                                     coco_json_path=coco_json_path,
                                     vocabulary=vocabulary,
                                     transform=transform, clean_annotations_data=clean_annotations_data)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    custom_data_loader = torch.utils.data.DataLoader(dataset=coco_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=shuffle,
                                                     num_workers=num_workers,
                                                     collate_fn=custom_collate_function)
    return custom_data_loader


def custom_collate_function(data_batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We need a custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data_batch: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data_batch.sort(key=lambda d: len(d[1]), reverse=True)
    imgs, captions = zip(*data_batch)

    # Merge images (from list of 3D tensors to 4D tensor).
    # Originally, imgs is a list of <batch_size> number of RGB images with dimensions (3, 256, 256)
    # This line of code turns it into a single tensor of dimensions (<batch_size>, 3, 256, 256)
    imgs = torch.stack(imgs, 0)

    # Merge captions (from list of 1D tensors to 2D tensor), similar to merging of images done above.
    caption_lengths: List[int] = [len(cap) for cap in captions]
    captions_tensor: torch.Tensor = torch.zeros(len(captions), max(caption_lengths)).long()
    for i, cap in enumerate(captions):
        end = len(cap)  # caption_lengths[i]
        captions_tensor[i, :end] = cap[:end]
    return imgs, captions_tensor, caption_lengths


def build_vocabulary(json_file_path: str, frequency_threshold_for_tokens: int):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json_file_path)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i + 1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i + 1, len(ids)))

    # special tokens
    tokens = ['<pad>', '<start>', '<end>', '<unk>']

    # If the word frequency is less than 'threshold', then the word is discarded.
    tokens.extend([token for token, cnt in counter.items() if cnt >= frequency_threshold_for_tokens])

    return Vocab(tokens=tokens)


class BasicLSTMBasedDecoderModel(nn.Module):
    def __init__(self, embedding_size, hidden_layer_size, vocabulary_size, num_layers, max_seq_len=20,
                 init_lstm_biases_to_one: bool = False):
        """Set the hyper-parameters and build the layers."""
        super(BasicLSTMBasedDecoderModel, self).__init__()
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm_layer = nn.LSTM(embedding_size, hidden_layer_size, num_layers, batch_first=True)
        if init_lstm_biases_to_one:
            init_lstm_forget_gate_biases(self.lstm_layer, value=1.0)
        self.linear_layer = nn.Linear(hidden_layer_size, vocabulary_size)
        self.max_seq_len = max_seq_len

    def forward(self, input_features, capts, lens):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embedding_layer(capts)
        embeddings = torch.cat((input_features.unsqueeze(1), embeddings), 1)
        lstm_input = pack_padded_sequence(embeddings, lens, batch_first=True)
        hidden_variables, _ = self.lstm_layer(lstm_input)
        model_outputs = self.linear_layer(hidden_variables[0])
        return model_outputs

    def sample(self, image_feature_tensor: torch.Tensor, lstm_states=None):
        """Generate captions for given image features using greedy search."""
        sampled_indices = []
        lstm_inputs = image_feature_tensor.unsqueeze(1)
        for i in range(self.max_seq_len):
            hidden_variables, lstm_states = self.lstm_layer(lstm_inputs,
                                                            lstm_states)  # hiddens: (batch_size, 1, hidden_size)
            model_outputs = self.linear_layer(hidden_variables.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted_word_index_batch = model_outputs.max(1)  # predicted: (batch_size)
            sampled_indices.append(predicted_word_index_batch)
            lstm_inputs = self.embedding_layer(predicted_word_index_batch)  # inputs: (batch_size, embed_size)
            lstm_inputs = lstm_inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_indices = torch.stack(sampled_indices, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_indices


class BasicEncoderDecoderModel(nn.Module):
    def __init__(self, encoder_output_size: int, decoder_hidden_layer_size: int, vocabulary_size: int,
                 summary_writer: SummaryWriter, resnet_model_type=ResNetModelType.RESNET34,
                 init_lstm_biases_to_one: bool = True):
        super(BasicEncoderDecoderModel, self).__init__()
        self.encoder_model = ResNetWithLastLayerModified(encoder_output_size,
                                                         resnet_model_type=resnet_model_type)
        self.decoder_model = BasicLSTMBasedDecoderModel(encoder_output_size, decoder_hidden_layer_size,
                                                        vocabulary_size, 1,
                                                        init_lstm_biases_to_one=init_lstm_biases_to_one)
        self.summary_writer = summary_writer

    def forward(self, input_images: torch.Tensor, captions: torch.Tensor, caption_lengths: torch.Tensor):
        image_features: torch.Tensor = self.encoder_model(input_images)

        self.summary_writer.add_histogram(tag='image_batch', values=input_images.data)
        self.summary_writer.add_histogram(tag='encoded_image_batch', values=image_features.data)

        return self.decoder_model(image_features, captions, caption_lengths)

    def sample(self, input_image_tensor: torch.Tensor, lstm_states=None):
        image_feature_tensor: torch.Tensor = self.encoder_model(input_image_tensor)
        return self.decoder_model.sample(image_feature_tensor, lstm_states)

    def get_trainable_parameters(self):
        return list(self.decoder_model.parameters()) + list(self.encoder_model.linear_layer.parameters()) + list(
            self.encoder_model.batch_norm.parameters())


class BasicEncoderDecoderModelWithAttention(nn.Module):
    def __init__(self, resnet_model_type: ResNetModelType, vocabulary_size: int, pretrained: bool = True,
                 freeze_pretrained_layers: bool = True, image_embedding_size: int = 512):
        super(BasicEncoderDecoderModelWithAttention, self).__init__()

        """Load the pretrained ResNet-X and replace top fc layer."""
        resnet = get_resnet_model(resnet_model_type, pretrained)

        module_list = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet_with_conv_output_as_last_layer: nn.Sequential = nn.Sequential(*module_list[:-1])
        self.avg_pooling_layer: AdaptiveAvgPool2d = module_list[-1]
        self.pretrained = pretrained
        self.freeze_pretrained_layers = freeze_pretrained_layers
        self.attention_layer = BidirectionalAttention(input_size=image_embedding_size,
                                                      return_with_inputs_concatenated=False,
                                                      scale_attention_weights_for_rhs=False,
                                                      scale_attention_weights_for_lhs=False)

        self.embedding_layer = nn.Embedding(vocabulary_size, image_embedding_size)

        self.lstm_decoder = torch.nn.LSTM(input_size=image_embedding_size * 2, hidden_size=image_embedding_size,
                                          batch_first=True, bias=False)
        self.final_linear_layer = nn.Linear(image_embedding_size, vocabulary_size)

    def forward(self, input_images: torch.Tensor, captions: torch.Tensor, caption_lengths: torch.Tensor):
        """Extract feature vectors from input images."""
        if self.pretrained and self.freeze_pretrained_layers:
            with torch.no_grad():
                resnet_conv_output: torch.Tensor = self.resnet_with_conv_output_as_last_layer(input_images)
                resnet_pooled_output: torch.Tensor = self.avg_pooling_layer(resnet_conv_output)
        else:
            resnet_conv_output: torch.Tensor = self.resnet_with_conv_output_as_last_layer(input_images)
            resnet_pooled_output: torch.Tensor = self.avg_pooling_layer(resnet_conv_output)

        # (N, num_filters, 1, 1) => (N, 1, num_filters)
        pooled_output_squeezed = torch.transpose(resnet_pooled_output.squeeze(dim=3), dim0=1, dim1=2)

        embeddings = torch.cat(tensors=(pooled_output_squeezed, self.embedding_layer(captions)), dim=1)
        resnet_conv_output = resnet_conv_output.reshape((resnet_conv_output.shape[0],
                                                         resnet_conv_output.shape[1],
                                                         resnet_conv_output.shape[2] * resnet_conv_output.shape[3], 1))
        resnet_conv_output = torch.transpose(resnet_conv_output.squeeze(dim=3), dim0=1, dim1=2)
        resnet_conv_output_as_weighted_embeddings, embeddings_as_weighted_conv_output = self.attention_layer(
            resnet_conv_output, embeddings)

        lstm_input = pack_padded_sequence(input=torch.cat(tensors=(embeddings,
                                                                   embeddings_as_weighted_conv_output), dim=2),
                                          lengths=caption_lengths, batch_first=True)
        hidden_states, _ = self.lstm_decoder(lstm_input)
        return self.final_linear_layer(hidden_states[0])
