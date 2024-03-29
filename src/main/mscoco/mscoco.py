# -*- coding: utf-8 -*-

import os
import sys
from enum import Enum
from typing import Tuple, List

import nltk
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from src.main.common.vocabulary import Vocab
from src.main.mscoco.utils_and_models import get_loader, build_vocabulary, BasicEncoderDecoderModelWithAttention, \
    BasicEncoderDecoderModel
from src.modules.common.resnet_models import ResNetModelType

dataset_data_file_path: str = "/Users/abhishek.saxena/Documents/personal/mscoco"


class RunMode(Enum):
    """
    Enum instances in decreasing order of dataset size
    """
    ACTUAL_DATA = "ACTUAL_DATA"
    SMALL_DATA = "SMALL_DATA"
    SAMPLE_DATA = "SAMPLE_DATA"


run_mode: RunMode = RunMode.SAMPLE_DATA
num_epochs: int = 1
num_dataloader_workers: int = 0
batch_size: int = 4

if not os.path.exists("/Users/abhishek.saxena"):
    nltk.download('punkt')

    sys.path.append('/content/gdrive/MyDrive/personal/ml_experiments')
    sys.path.append('/content/gdrive/MyDrive/personal/datasets')

    from google.colab import drive

    drive.mount("/content/gdrive")
    dataset_data_file_path: str = "/content/gdrive/MyDrive/personal/datasets/mscoco"

    run_mode: RunMode = RunMode.SMALL_DATA
    num_epochs: int = 4
    num_dataloader_workers: int = 0
    batch_size: int = 128


# note: these will be used if run_mode == RunMode.ACTUAL. The actual datasets haven't been downsized yet
training_data_path: str = dataset_data_file_path + "/train2017"
validation_data_path: str = dataset_data_file_path + "/val2017"
training_annotations_file_path: str = dataset_data_file_path + "/annotations/captions_train2017.json"
validation_annotations_file_path: str = dataset_data_file_path + "/annotations/captions_val2017.json"

if run_mode == RunMode.SAMPLE_DATA:
    training_annotations_file_path: str = dataset_data_file_path + "/annotations/captions_val2017.json"
    validation_annotations_file_path: str = dataset_data_file_path + "/annotations/captions_val2017.json"
    training_data_path: str = dataset_data_file_path + "/val2017_downsized_sample"
    validation_data_path: str = dataset_data_file_path + "/val2017_downsized_sample"
elif run_mode == RunMode.SMALL_DATA:
    # yep, turning the tables here a bit. Training on a smaller (validation) dataset and testing on a
    # sample of training data
    training_annotations_file_path: str = dataset_data_file_path + "/annotations/captions_val2017.json"
    validation_annotations_file_path: str = dataset_data_file_path + "/annotations/captions_train2017.json"
    training_data_path: str = dataset_data_file_path + "/val2017_downsized"
    validation_data_path: str = dataset_data_file_path + "/train2017_sample_downsized"

vocabulary: Vocab = build_vocabulary(json_file_path=training_annotations_file_path, frequency_threshold_for_tokens=2)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model directory
if not os.path.exists('models_dir/'):
    os.makedirs('models_dir/')

# channelwise_image_means_for_normalization: Tuple = (0.485, 0.456, 0.406)
# channelwise_image_std_for_normalization: Tuple = (0.229, 0.224, 0.225)

channelwise_image_means_for_normalization: Tuple = (0.0, 0.0, 0.0)
channelwise_image_std_for_normalization: Tuple = (1, 1, 1)

# Image preprocessing, normalization for the pretrained resnet
transforms_to_apply = transforms.Compose([
    transforms.RandomCrop(125),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(channelwise_image_means_for_normalization, channelwise_image_std_for_normalization)])

# Build data loader
custom_data_loader = get_loader(data_path=training_data_path, coco_json_path=training_annotations_file_path,
                                vocabulary=vocabulary, transform=transforms_to_apply,
                                batch_size=batch_size, shuffle=True, num_workers=num_dataloader_workers,
                                clean_annotations_data=True)

summary_writer: SummaryWriter = SummaryWriter()
vocabulary_size: int = len(vocabulary)

# Build the models
basic_encoder_decoder = BasicEncoderDecoderModel(encoder_output_size=128, decoder_hidden_layer_size=256,
                                                 vocabulary_size=vocabulary_size,
                                                 summary_writer=summary_writer,
                                                 resnet_model_type=ResNetModelType.RESNET34).to(device)

attention_based_encoder_decoder = BasicEncoderDecoderModelWithAttention(resnet_model_type=ResNetModelType.RESNET18,
                                                                        vocabulary_size=vocabulary_size).to(device)

# Loss and optimizer
loss_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(basic_encoder_decoder.get_trainable_parameters(), lr=0.001)

# Train the models
total_num_steps: int = len(custom_data_loader)

for epoch in tqdm(range(num_epochs)):
    for i, (images, captions, caption_lengths) in enumerate(custom_data_loader):

        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, caption_lengths, batch_first=True)[0]

        # Forward, backward and optimize
        # aux_output: torch.Tensor = attention_based_encoder_decoder(input_images=images, captions=captions,
        #                                                            caption_lengths=caption_lengths)

        outputs = basic_encoder_decoder(images, captions, caption_lengths)

        loss = loss_criterion(outputs, targets)

        basic_encoder_decoder.zero_grad()
        loss.backward()
        optimizer.step()

        # Print log info
        if i % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                  .format(epoch, num_epochs, i, total_num_steps, loss.item(), np.exp(loss.item())))

        # Save the model checkpoints
        if (i + 1) % 2000 == 0:
            torch.save(basic_encoder_decoder.state_dict(), os.path.join(
                'models_dir/', 'basic_encoder_decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))


basic_encoder_decoder.eval()

validation_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(channelwise_image_means_for_normalization,
                         channelwise_image_std_for_normalization)])

validation_image_files: List[str] = os.listdir(validation_data_path)

with torch.no_grad():
    for img_file in validation_image_files:
        if not img_file.endswith('.jpg'):
            continue

        validation_file_path = os.path.join(validation_data_path, img_file)
        image_tensor: torch.Tensor = validation_transforms(Image.open(validation_file_path).convert('RGB')).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        sampled_caption_indices = basic_encoder_decoder.sample(image_tensor)
        sampled_caption_indices = sampled_caption_indices[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)

        # Convert word_ids to words
        predicted_caption_tokens = []

        for token_index in sampled_caption_indices:
            word = vocabulary.index_to_word[token_index]
            predicted_caption_tokens.append(word)
            if word == '<end>':
                break
        print("image file: " + img_file)
        print(" ".join(predicted_caption_tokens))

summary_writer.close()
