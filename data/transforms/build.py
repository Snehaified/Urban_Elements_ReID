import random
import torchvision.transforms as T

from .transforms import *
from .autoaugment import AutoAugment
from PIL import Image, ImageFilter, ImageOps

from .transforms import LGT

import torchvision.transforms.functional as F
from PIL import Image

class PadToAspect(object):
    """
    Pads a PIL image to match a target aspect ratio (target_w / target_h)
    using constant value padding, preventing geometric distortion during Resize.
    """
    def __init__(self, target_h, target_w, fill=(127, 127, 127)):
        self.target_h = target_h
        self.target_w = target_w
        self.target_ratio = target_w / target_h
        # 127 is roughly the ImageNet mean, perfect for a neutral background
        self.fill = fill 

    def __call__(self, img):
        w, h = img.size
        img_ratio = w / max(h, 1) # Prevent division by zero

        if abs(img_ratio - self.target_ratio) < 1e-5:
            return img

        # If the image is proportionally wider than the target (e.g., Crosswalks)
        # We need to pad the top and bottom
        if img_ratio > self.target_ratio:
            new_h = int(w / self.target_ratio)
            pad_h = new_h - h
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            # padding = (left, top, right, bottom)
            padding = (0, pad_top, 0, pad_bottom)
            
        # If the image is proportionally taller than the target (e.g., Rubbish Bins)
        # We need to pad the left and right
        else:
            new_w = int(h * self.target_ratio)
            pad_w = new_w - w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            padding = (pad_left, 0, pad_right, 0)

        return F.pad(img, padding, self.fill, 'constant')

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

def build_transforms(cfg, is_train=True, is_fake=False):
    res = []

    if is_train:
        size_train = cfg.INPUT.SIZE_TRAIN

        # augmix augmentation
        do_augmix = cfg.INPUT.DO_AUGMIX

        # auto augmentation
        do_autoaug = cfg.INPUT.DO_AUTOAUG
        # total_iter = cfg.SOLVER.MAX_ITER
        total_iter = cfg.SOLVER.MAX_EPOCHS

        # horizontal filp
        do_flip = cfg.INPUT.DO_FLIP
        flip_prob = cfg.INPUT.FLIP_PROB

        # padding
        do_pad = cfg.INPUT.DO_PAD
        padding = cfg.INPUT.PADDING
        padding_mode = cfg.INPUT.PADDING_MODE

        # Local Grayscale Transfomation
        do_lgt = cfg.INPUT.LGT.DO_LGT
        lgt_prob = cfg.INPUT.LGT.PROB

        # color jitter
        do_cj = cfg.INPUT.CJ.ENABLED
        cj_prob = cfg.INPUT.CJ.PROB
        cj_brightness = cfg.INPUT.CJ.BRIGHTNESS
        cj_contrast = cfg.INPUT.CJ.CONTRAST
        cj_saturation = cfg.INPUT.CJ.SATURATION
        cj_hue = cfg.INPUT.CJ.HUE

        # random erasing
        do_rea = cfg.INPUT.REA.ENABLED
        rea_prob = cfg.INPUT.REA.PROB
        rea_mean = cfg.INPUT.REA.MEAN
        # random patch
        do_rpt = cfg.INPUT.RPT.ENABLED
        rpt_prob = cfg.INPUT.RPT.PROB

        if do_autoaug:
            res.append(AutoAugment(total_iter))
        if cfg.INPUT.ASPECT_PADDING:
            res.append(PadToAspect(target_h=size_train[0], target_w=size_train[1]))
        res.append(T.Resize(size_train, interpolation=3))
        if cfg.INPUT.PERSPECTIVE.ENABLED:
            res.append(T.RandomPerspective(distortion_scale=cfg.INPUT.PERSPECTIVE.DISTORTION,
                                           p=cfg.INPUT.PERSPECTIVE.PROB))
        if cfg.INPUT.ROTATION.ENABLED:
            res.append(T.RandomRotation(degrees=cfg.INPUT.ROTATION.DEGREES))
        if do_flip:
            res.append(T.RandomHorizontalFlip(p=flip_prob))
        if do_pad:
            res.extend([T.Pad(padding, padding_mode=padding_mode),
                        T.RandomCrop(size_train)])
        if do_lgt:
            res.append(LGT(lgt_prob))
        if do_cj:
            res.append(T.RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=cj_prob))
        if do_augmix:
            res.append(AugMix())
        # if do_rea:
        #     res.append(RandomErasing(probability=rea_prob, mean=rea_mean, sh=1/3))
        if do_rpt:
            res.append(RandomPatch(prob_happen=rpt_prob))
        if is_fake:
            if cfg.META.DATA.SYNTH_FLAG == 'jitter':
                res.append(T.RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=1.0))
            elif cfg.META.DATA.SYNTH_FLAG == 'augmix':
                res.append(AugMix())
            elif cfg.META.DATA.SYNTH_FLAG == 'both':
                res.append(T.RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=cj_prob))
                res.append(AugMix())
        res.extend([
            T.ToTensor(),
            T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        if do_rea:
            from timm.data.random_erasing import RandomErasing as RE
            res.append(RE(probability=rea_prob, mode='pixel', max_count=1, device='cpu'))
    else:
        size_test = cfg.INPUT.SIZE_TEST
        if cfg.INPUT.ASPECT_PADDING:
            res.append(PadToAspect(target_h=size_test[0], target_w=size_test[1]))
        res.append(T.Resize(size_test, interpolation=3))
        res.extend([
            T.ToTensor(),
            T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
    return T.Compose(res)
