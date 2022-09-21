import argparse, os, sys, glob
import logging

import PIL
import cv2
import torch
import numpy as np
import asyncio
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
import time
from torch import autocast
from contextlib import contextmanager, nullcontext
from generate_request import GenerateRequest
from ldm import data

# from pytorch_lightning import seed_everything

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor


class SDGenerator:

    @staticmethod
    def _get_device():
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    @staticmethod
    def load_replacement(self, x):
        try:
            hwc = x.shape
            y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
            y = (np.array(y) / 255.0).astype(x.dtype)
            assert y.shape == x.shape
            return y
        except Exception:
            return x

    @staticmethod
    def _check_safety(self, x_image):
        return x_image, False

    @staticmethod
    def _put_watermark(img, wm_encoder=None):
        if wm_encoder is not None:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = wm_encoder.encode(img, 'dwtDct')
            img = Image.fromarray(img[:, :, ::-1])
        return img

    @staticmethod
    def _numpy_to_pil(self, images):
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    @staticmethod
    def _load_img(self, path):
        image = Image.open(path).convert("RGB")
        w, h = image.size
        print(f"loaded input image of size ({w}, {h}) from {path}")
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2. * image - 1.

    @staticmethod
    def _load_model_from_config(self, config, ckpt, verbose=False):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

        model.to(self._get_device())
        model.eval()
        return model

    def __init__(self, params):
        self._device = torch.device(self._get_device())
        self._out_dir = params.get("out_dir", "tmp")
        self._skip_grid = params.get("skip_grid", True)
        self._skip_save = params.get("skip_save", False)
        self._config_file = params.get("config_file", "configs/stable-diffusion/v1-inference.yaml")
        self._ckpt = params.get("ckpt", "models/ldm/stable-diffusion-v1/model.ckpt")
        self._config = OmegaConf.load(f"{self._config_file}")
        self._model = self._load_model_from_config(self=self, config=self._config, ckpt=f"{self._ckpt}")
        self._model = self._model.to(self._device)
        self._plms = params.get("plms", False)
        self._precision_scope = params.get("precision_scope", "autocast")
        self._do_safety_check = params.get("do_safety_check", False)
        if self._plms:
            raise NotImplementedError("PLMS sampler not (yet) supported")
            self._sampler = PLMSSampler(self._model)
        else:
            self._sampler = DDIMSampler(self._model)
        os.makedirs(self._out_dir, exist_ok=True)
        self._wm = "StableDiffusionV1"
        self._wm_encoder = WatermarkEncoder()
        self._wm_encoder.set_watermark('bytes', self._wm.encode('utf-8'))
        safety_model_id = "CompVis/stable-diffusion-safety-checker"
        safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    def generate_image(self, req: GenerateRequest):
        init_image = None
        init_latent = None
        if None != req.initial_image_path:
            print("loading image {}".format(req.initial_image_path))
            # continue later
            assert os.path.isfile(req.initial_image_path)
            init_image = self._load_img(req.initial_image_path).to(self._device)
            init_image = repeat(init_image, '1 ... -> b ...', b=req.count)
            init_latent = self._model.get_first_stage_encoding(self._model.encode_first_stage(init_image))
            t_enc = int(req.strength * req.ddim_steps)
            print(f"target t_enc is {t_enc} steps")

        seed_everything(req.seed)
        self._sampler.make_schedule(ddim_num_steps=req.ddim_steps, ddim_eta=req.ddim_eta, verbose=req.verbose)
        if self._device.type == 'mps':
            precision_scope = nullcontext

        data = [req.count * req.prompt]
        seed_everything(req.seed)
        with torch.no_grad():
            with precision_scope(self._device.type):
                with self._model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in trange(req.count, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if req.scale != 1.0:
                                uc = self._model.get_learned_conditioning(req.count * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self._model.get_learned_conditioning(prompts)
                            shape = [req.channels, req.height // req.down_sample_factor,
                                     req.width // req.down_sample_factor]
                            samples_ddim, _ = self._sampler.sample(S=req.ddim_steps,
                                                                   conditioning=c,
                                                                   batch_size=req.count,
                                                                   shape=shape,
                                                                   verbose=False,
                                                                   unconditional_guidance_scale=req.scale,
                                                                   unconditional_conditioning=uc,
                                                                   eta=req.ddim_eta,
                                                                   x_T=req.start_code)

                            x_samples_ddim = self._model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                            x_checked_image, has_nsfw_concept = self._check_safety(self=self, x_image=x_samples_ddim)

                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                            if not self._skip_save:
                                for x_sample in x_checked_image_torch:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))
                                    img = self._put_watermark(img, self._wm_encoder)
                                    img.save(os.path.join(self._out_dir, f"{req.image_id}.png"))

    def get_image(self, image_id):
        file_name = os.path.join(self._out_dir, f"{image_id}.png")
        logging.info("Fetching:{}".format(file_name))
        return Image.open(file_name)
