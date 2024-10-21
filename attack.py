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

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


def classifier_preprocess(test_image):
    test_image = test_image.resize((224, 224), resample=Image.LANCZOS)
    test_image = np.float32(test_image) / 255.0
    test_image = test_image[:, :, :3]
    test_image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    test_image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    test_image = test_image.transpose((2, 0, 1))
    test_image = torch.from_numpy(test_image).unsqueeze(0)
    return test_image.cuda()


def instrp2p_preprocess(image, resolution):
    width, height = image.size
    factor = resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    return input_image


def output_classifier_preprocess(x):
    x = (x / 2 + 0.5).clamp(0, 1)
    x = F.interpolate(x, size=224, mode='area')
    test_image = x.permute(0, 2, 3, 1)
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=test_image.dtype, device=test_image.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=test_image.dtype, device=test_image.device)
    test_image = test_image[:, :, :].sub(mean).div(std)
    test_image = test_image.permute(0, 3, 1, 2)
    # test_image = F.interpolate(test_image, size=224, mode='area')
    return test_image


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
        pil_img.save(save_path)


def inference(model, data):
    intermediate_features = []

    def hook_fn(module, input, output):
        intermediate_features.append(input[0])

    hook_handle = model.avgpool.register_forward_hook(hook_fn)
    output = model(data)
    hook_handle.remove()
    return output, intermediate_features


def get_and_show_attention(features, view=False):
    first_dimension_size = features.shape[0]
    attention_map = features.sum(dim=0) / first_dimension_size
    if view:
        image = 255 * attention_map / attention_map.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.detach().cpu().numpy().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        view_images(image, show=True)
    return attention_map


def show_latent(latent, model):
    x = model.decode_first_stage(latent).requires_grad_(True)
    edited_image = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
    edited_image = 255.0 * rearrange(edited_image, "1 c h w -> h w c")
    edited_image = Image.fromarray(edited_image.type(torch.uint8).cpu().numpy())
    edited_image = ImageOps.fit(edited_image, (299, 299), method=Image.Resampling.LANCZOS)
    edited_image.show()
    return 0


def main():
    parser = ArgumentParser()
    parser.add_argument("--learning_rate", default=1e-1, type=float)
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=20, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)

    parser.add_argument('--start_step', default=16, type=int, help='Which step to start the attack')
    parser.add_argument('--images_root', default="data/demo_paper/images", type=str,
                        help='The clean images root directory')
    parser.add_argument('--label_path', default="data/demo_paper/labels.txt", type=str,
                        help='The clean images labels.txt')
    parser.add_argument('--save_dir', default="output", type=str,
                        help='Where to save the adversarial examples, and other results')
    parser.add_argument('--iterations', default=20, type=int, help='Iterations of optimizing the adv_image')
    parser.add_argument('--classifier_name', default="inception", type=str,
                        help='The surrogate model from which the adversarial examples are crafted')
    parser.add_argument('--dataset_name', default="imagenet_compatible", type=str,
                        choices=["imagenet_compatible", "cub_200_2011", "stanford_car"],
                        help='The dataset name for generating adversarial examples')
    parser.add_argument("--verbose", default=1, type=bool)

    parser.add_argument("--edit", default="make it in fog", type=str)
    parser.add_argument("--cfg_text", default=7.5, type=float)
    parser.add_argument("--cfg_image", default=2, type=float)
    parser.add_argument("--lamda1", default=0.001, type=float)
    parser.add_argument("--lamda2", default=0.1, type=float)
    parser.add_argument("--seed", default=100, type=int)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    if args.dataset_name == "imagenet_compatible":
        assert args.classifier_name not in ["cubResnet50", "cubSEResnet154", "cubSEResnet101", "carResnet50", "carSEResnet154",
                                  "carSEResnet101"], f"There is no pretrained weight of {args.classifier_name} for ImageNet-Compatible dataset."
    if args.dataset_name == "cub_200_2011":
        assert args.classifier_name in ["cubResnet50", "cubSEResnet154",
                              "cubSEResnet101"], f"There is no pretrained weight of {args.classifier_name} for CUB_200_2011 dataset."
    if args.dataset_name == "standford_car":
        assert args.classifier_name in ["carResnet50", "carSEResnet154",
                              "carSEResnet101"], f"There is no pretrained weight of {args.classifier_name} for Standford Cars dataset."

    save_dir = f"{args.save_dir}_{args.steps}_{args.start_step}_{args.iterations}_{args.edit}_{args.cfg_text}_{args.cfg_image}_{args.lamda1}_{args.lamda2}"  # Where to save the adversarial examples, and other results.
    os.makedirs(save_dir, exist_ok=True)

    images_root = args.images_root  # The clean images' root directory.
    label_path = args.label_path  # The clean images' labels.txt.
    with open(label_path, "r") as f:
        label = []
        for i in f.readlines():
            # label.append(int(i.rstrip()))
            label.append(int(i.rstrip()) - 1)  # The label number of the imagenet-compatible dataset starts from 1.
        label = np.array(label)

    print(f"\n******Attack based on instruct-pix2pix, Attacked Dataset: {args.dataset_name}*********")

    "Attack a subset images"
    all_images = glob.glob(os.path.join(images_root, "*"))
    all_images = natsorted(all_images, alg=ns.PATH)

    adv_images = []
    images = []

    classifier = other_attacks.model_selection(args.classifier_name).eval()
    classifier.requires_grad_(False)

    for ind, image_path in enumerate(all_images):
        tmp_image = Image.open(image_path).convert('RGB')
        tmp_image.save(os.path.join(save_dir, str(ind).rjust(4, '0') + "_originImage.png"))

        seed = random.randint(0, 100000) if args.seed is None else args.seed
        input_image = instrp2p_preprocess(tmp_image, args.resolution)

        test_image = classifier_preprocess(tmp_image)
        gt_label = torch.from_numpy(label[ind:ind + 1]).long().cuda()

        before_pred, before_features = inference(classifier, test_image.cuda())
        logit = torch.nn.Softmax()(before_pred)
        print("gt_label:", gt_label[0].item(), "pred_gt_logit:",
          logit[0, gt_label[0]].item(), "pred_label:", torch.argmax(before_pred, 1).detach().item())

        # get and show attention map
        dim = before_features[0][0].shape[0]
        height = before_features[0][0].shape[1]
        width = before_features[0][0].shape[2]
        attention_map = get_and_show_attention(before_features[0][0], True).expand(dim, height, width)

        with torch.no_grad(), autocast("cuda"), model.ema_scope():
            cond = {}
            cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])]
            input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
            input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
            cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

            uncond = {}
            uncond["c_crossattn"] = [null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            sigmas = model_wrap.get_sigmas(args.steps)
            sliced_sigmas1 = sigmas[0:args.start_step]
            sliced_sigmas2 = sigmas[args.start_step-1:]

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": args.cfg_text,
                "image_cfg_scale": args.cfg_image,
            }
            torch.manual_seed(seed)
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            # show_latent(z, model)
            z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sliced_sigmas1, extra_args=extra_args)
            # show_latent(z, model)
            # x = model.decode_first_stage(z).requires_grad_(True)
            # edited_image = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            # edited_image = 255.0 * rearrange(edited_image, "1 c h w -> h w c")
            # edited_image = Image.fromarray(edited_image.type(torch.uint8).cpu().numpy())
            # edited_image = ImageOps.fit(edited_image, (299, 299), method=Image.Resampling.LANCZOS)
            # edited_image.show()
        optimizer = optim.AdamW([z], lr=args.learning_rate)
        cross_entro = torch.nn.CrossEntropyLoss()
        pbar = tqdm(range(args.iterations), desc="Optimize Iterations")
        for _, _ in enumerate(pbar):
            with torch.set_grad_enabled(True):
                z = z.to(torch.float32).requires_grad_(True)
                z_0 = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas[args.start_step - 1:], extra_args=extra_args)
                x = model.decode_first_stage(z_0)
                out_image = output_classifier_preprocess(x)

                # edited_image = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                # edited_image = 255.0 * rearrange(edited_image, "1 c h w -> h w c")
                # edited_image = Image.fromarray(edited_image.type(torch.uint8).cpu().numpy())
                # edited_image = ImageOps.fit(edited_image, (299, 299), method=Image.Resampling.LANCZOS)
                # edited_image.show()

                pred, after_features = inference(classifier, out_image.cuda())

                if args.dataset_name != "imagenet_compatible":
                    pred = pred / 10
                    # pred = classifier(out_image) / 10
                # else:
                #     pred = classifier(out_image)

                after_logit = torch.nn.Softmax()(pred)
                after_pred_label = torch.argmax(pred, 1).detach().item()

                attack_loss = - cross_entro(pred, gt_label)
                L2 = LpDistance(2)
                distance = L2(before_features[0], after_features[0])
                similarity_loss = args.lamda1 * distance

                before_att_feature = torch.mul(attention_map, before_features[0][0]).reshape(-1)
                after_att_feature = torch.mul(attention_map, after_features[0][0]).reshape(-1)
                attention_loss = args.lamda2 * F.cosine_similarity(before_att_feature, after_att_feature, dim=0)

                loss = attack_loss + similarity_loss + attention_loss

                if args.verbose:
                    pbar.set_postfix_str(
                        f"attack_loss: {attack_loss.item():.5f} "
                        f"similarity_loss: {similarity_loss.item():.5f} "
                        f"attention_loss: {attention_loss.item():.5f} "
                        f"after_pred_label: {after_pred_label:d} "
                        f"after_gt_logit: {after_logit[0, gt_label[0]].item():.16f} "
                        f"loss: {loss.item():.5f} ")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            z_0 = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas[args.start_step - 1:], extra_args=extra_args)
            # show_latent(z, model)
            x = model.decode_first_stage(z_0)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = F.interpolate(x, size=224, mode='area')
            perturbed = rearrange(x, "1 c h w ->1 h w c").cpu().numpy()

            init_image = tmp_image.resize((224, 224), resample=Image.LANCZOS)
            real = np.array(init_image).astype(np.float32)/255
            real = np.expand_dims(real, axis=0)

            save_path = os.path.join(save_dir, str(ind).rjust(4, '0'))
            view_images(np.concatenate([real, perturbed]) * 255, show=False,
                        save_path=save_path + "_diff_{}_image_{}.png".format(args.classifier_name,
                                                                             "ATKSuccess"
                                                                             if after_pred_label != gt_label[0].item()
                                                                             else "Fail"))
            view_images(perturbed * 255, show=False, save_path=save_path + "_adv_image.png")

            L1 = LpDistance(1)
            L2 = LpDistance(2)
            Linf = LpDistance(float("inf"))

            print("L1: {}\tL2: {}\tLinf: {}".format(L1(real, perturbed), L2(real, perturbed), Linf(real, perturbed)))

            diff = perturbed - real
            diff = (diff - diff.min()) / (diff.max() - diff.min()) * 255

            view_images(diff.clip(0, 255), show=False,
                        save_path=save_path + "_diff_relative.png")

            diff = (np.abs(perturbed - real) * 255).astype(np.uint8)
            view_images(diff.clip(0, 255), show=False,
                        save_path=save_path + "_diff_absolute.png")

            perturbed = perturbed[0]
            real = real[0]
            adv_images.append(perturbed[None].transpose(0, 3, 1, 2))
            images.append(real[None].transpose(0, 3, 1, 2))

    images = np.concatenate(images)
    adv_images = np.concatenate(adv_images)

    """
            Test the robustness of the generated adversarial examples across a variety of normally trained models or
            adversarially trained models.
    """
    model_transfer(images, adv_images, label, 224, save_path=save_dir, args=args, niqe=0, brisque=0)


if __name__ == "__main__":
    main()
