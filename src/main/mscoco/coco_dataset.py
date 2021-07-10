import os
from typing import Callable

import nltk
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils import data as data
from torchvision import transforms

from src.main.common.vocabulary import Vocab


class CustomCocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, data_path: str, coco_json_path: str, vocabulary: Vocab,
                 transform: Callable = None, clean_annotations_data: bool = False):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            data_path: image directory.
            coco_json_path: coco annotation file path.
            vocabulary: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = data_path
        self.coco_data = COCO(coco_json_path)
        self.vocabulary = vocabulary
        self.transform = transform if transform else transforms.ToTensor()

        # this flag was created to deal with the issue that we might wanna experiment
        # with a much smaller sample of images, however, since it was hard to split the filter the
        # captions annotations file to limit it to the sample dataset, we chose to clean the
        # annotation_indices object instead
        all_image_paths = []
        if clean_annotations_data:
            cleaned_annotations = {}
            for annotation_id in self.coco_data.anns.keys():
                image_path = self.__get_image_path_and_caption(annotation_id)[0]
                if os.path.exists(image_path):
                    cleaned_annotations[annotation_id] = self.coco_data.anns[annotation_id]
                    all_image_paths.append(image_path)

            self.coco_data.anns = cleaned_annotations
            print("all image paths len: " + str(len(all_image_paths)))
            print("unique image paths len: " + str(len(set(all_image_paths))))

        self.annotation_ids = list(self.coco_data.anns.keys())
        for id in self.annotation_ids:
            caption = self.coco_data.anns[id]['caption']
            # temporary hack for checking certain phrases in the dataset
            # if 'red and white' in caption:
            #     print("found text")
            #     print("annotation id = " + str(id))
            #     print("caption = " + caption)
            #     print("image = " + self.coco_data.loadImgs(self.coco_data.anns[id]['image_id'])[0]['file_name'])

    def __get_image_path_and_caption(self, annotation_id):
        image_id = self.coco_data.anns[annotation_id]['image_id']
        image_filename = self.coco_data.loadImgs(image_id)[0]['file_name']
        caption = self.coco_data.anns[annotation_id]['caption']

        return os.path.join(self.root, image_filename), caption

    def __getitem__(self, idx):
        """Returns one data pair (image and caption)."""
        annotation_id = self.annotation_ids[idx]
        image_path, caption = self.__get_image_path_and_caption(annotation_id)
        image = self.transform(Image.open(image_path).convert('RGB'))

        # Convert caption (string) to word ids.
        word_tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = [self.vocabulary('<start>')]
        caption.extend([self.vocabulary(token) for token in word_tokens])
        caption.append(self.vocabulary('<end>'))
        caption_word_indices_tensor = torch.Tensor(caption)
        return image, caption_word_indices_tensor

    def __len__(self):
        return len(self.annotation_ids)
