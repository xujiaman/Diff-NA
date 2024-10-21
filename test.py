from __future__ import annotations
import os
from natsort import ns, natsorted
import glob
from other_attacks import model_transfer
import numpy as np
from argparse import ArgumentParser
from PIL import Image
import pyiqa
import torch


def main():
    parser = ArgumentParser()
    parser.add_argument('--images_root', default="output", type=str,
                        help='')
    parser.add_argument('--save_dir', default="output/log", type=str,
                        help='')
    parser.add_argument('--label_path', default="data/imagenet_compatible/labels.txt", type=str,
                        help='The clean images labels.txt')
    parser.add_argument('--res', default=224, type=int, help='Input image resized resolution')
    parser.add_argument('--dataset_name', default="imagenet_compatible", type=str,
                        choices=["imagenet_compatible", "cub_200_2011", "stanford_car"],
                        help='The dataset name for generating adversarial examples')
    args = parser.parse_args()

    res = args.res
    save_dir = args.save_dir
    images_root = args.images_root
    label_path = args.label_path  # The clean images' labels.txt.

    os.makedirs(save_dir, exist_ok=True)

    with open(label_path, "r") as f:
        label = []
        for i in f.readlines():
            label.append(int(i.rstrip()) - 1)  # The label number of the imagenet-compatible dataset starts from 1.
        label = np.array(label)

    adv_images = []
    images = []
    #
    all_clean_images = glob.glob(os.path.join(images_root, "*originImage*"))
    # all_clean_images = glob.glob(os.path.join(images_root, "*"))
    all_clean_images = natsorted(all_clean_images, alg=ns.PATH)
    all_adv_images = glob.glob(os.path.join(images_root, "*adv_image*"))
    # all_adv_images = glob.glob(os.path.join(images_root, "*"))
    all_adv_images = natsorted(all_adv_images, alg=ns.PATH)
    niqe_values = []
    brisque_values = []
    niqe_metric = pyiqa.create_metric('niqe')
    brisque_metric = pyiqa.create_metric('brisque')

    for image_path, adv_image_path in zip(all_clean_images, all_adv_images):
    for adv_image_path in all_adv_images:
        tmp_image = Image.open(image_path).convert('RGB')
        tmp_image = tmp_image.resize((res, res), resample=Image.LANCZOS)
        tmp_image = np.array(tmp_image).astype(np.float32) / 255.0
        tmp_image = tmp_image[None].transpose(0, 3, 1, 2)
        images.append(tmp_image)

        tmp_image = Image.open(adv_image_path).convert('RGB')
        tmp_image = tmp_image.resize((res, res), resample=Image.LANCZOS)
        tmp_image = np.array(tmp_image).astype(np.float32) / 255.0
        tmp_image = tmp_image[None].transpose(0, 3, 1, 2)
        adv_images.append(tmp_image)

        nscore_fr = niqe_metric(adv_image_path)
        bscore_fr = brisque_metric(adv_image_path)

        niqe_values.append(nscore_fr)
        brisque_values.append(bscore_fr)

    images = np.concatenate(images)
    adv_images = np.concatenate(adv_images)

    average_niqe = sum(niqe_values) / len(niqe_values)
    average_brisque = sum(brisque_values) / len(brisque_values)

    """
            Test the robustness of the generated adversarial examples across a variety of normally trained models or
            adversarially trained models.
    """
    model_transfer(images, adv_images, label, res, save_path=save_dir, fid_path=images_root, args=args, niqe=average_niqe, brisque=average_brisque)
    # log = open(os.path.join(save_dir, "log.txt"), mode="w", encoding="utf-8")
    # print("\n*********niqe: {}********".format(average_niqe), file=log)
    # print("\n*********brisque: {}********".format(average_brisque), file=log)
    # log.close()


if __name__ == "__main__":
    main()