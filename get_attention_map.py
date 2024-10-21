from __future__ import annotations

import math
import random
import sys
from argparse import ArgumentParser
import os
from natsort import ns, natsorted
import glob
from tqdm import tqdm
import other_attacks
from torch import optim
import torch.nn.functional as F
from distances import LpDistance
from other_attacks import model_transfer
import cv2

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast


def classifier_preprocess(test_image):
    test_image = test_image.resize((224, 224), resample=Image.LANCZOS)
    test_image = np.float32(test_image) / 255.0
    test_image = test_image[:, :, :3]
    test_image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    test_image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    test_image = test_image.transpose((2, 0, 1))
    test_image = torch.from_numpy(test_image).unsqueeze(0)
    return test_image.cuda()


def view_images(images, num_rows=1, offset_ratio=0.02, save_path=None, show=False):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if show:
        pil_img.show()
    if save_path is not None:
        pil_img.save(save_path + "_attention_map.png")


def inference(model, data):
    intermediate_features = []

    def hook_fn(module, input, output):
        intermediate_features.append(input[0])

    hook_handle = model.avgpool.register_forward_hook(hook_fn)
    output = model(data)
    hook_handle.remove()
    return output, intermediate_features


def get_and_show_attention(features, save_path):
    first_dimension_size = features.shape[0]
    attention_map = features.sum(dim=0) / first_dimension_size
    image = 255 * attention_map / attention_map.max()
    image = image.unsqueeze(-1).expand(*image.shape, 3)
    image = image.detach().cpu().numpy().astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = np.array(Image.fromarray(image).resize((256, 256)))
    view_images(image, save_path=save_path, show=False)
    return attention_map


def main():
    parser = ArgumentParser()
    parser.add_argument('--images_root', default="data/test100/images", type=str,
                        help='The clean images root directory')
    parser.add_argument('--classifier_name', default="resnet", type=str,
                        help='The surrogate model from which the adversarial examples are crafted')
    parser.add_argument('--save_dir', default="data/test100/output_attention_map", type=str,
                        help='Where to save the adversarial examples, and other results')
    args = parser.parse_args()

    save_dir = args.save_dir  # Where to save the attention maps.
    os.makedirs(save_dir, exist_ok=True)
    images_root = args.images_root  # The clean images' root directory.
    all_images = glob.glob(os.path.join(images_root, "*"))
    all_images = natsorted(all_images, alg=ns.PATH)

    for ind, image_path in enumerate(all_images):
        save_path = os.path.join(save_dir, str(ind).rjust(4, '0'))
        tmp_image = Image.open(image_path).convert('RGB')
        test_image = classifier_preprocess(tmp_image)
        classifier = other_attacks.model_selection(args.classifier_name).eval()
        classifier.requires_grad_(False)

        before_pred, before_features = inference(classifier, test_image.cuda())

        # get and show attention map
        get_and_show_attention(before_features[0][0], save_path)


if __name__ == "__main__":
    main()