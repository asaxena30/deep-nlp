import os
from typing import Tuple

from PIL import Image


def reshape_images_and_copy_to_target(source_path: str, target_path: str, shape: Tuple[int, int]):
    images = os.listdir(source_path)
    num_im = len(images)
    for i, im in enumerate(images):
        if not im.endswith(".jpg"):
            continue
        with open(os.path.join(source_path, im), 'r+b') as f:
            with Image.open(f) as image:
                new_img = image.resize(shape, Image.ANTIALIAS)
                new_img.save(os.path.join(target_path, im), image.format)
        if (i + 1) % 100 == 0:
            print("[{}/{}] Resized the images and saved into '{}'."
                  .format(i + 1, num_im, target_path))


image_source_path = "/Users/abhishek.saxena/Documents/personal/mscoco/train2017_sample"
image_target_path = "/Users/abhishek.saxena/Documents/personal/mscoco/train2017_sample_downsized_2"

if not os.path.exists(image_target_path):
    os.mkdir(image_target_path)

reshape_images_and_copy_to_target(image_source_path, image_target_path, (128, 128))




