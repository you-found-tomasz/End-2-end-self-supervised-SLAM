import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import os
import torch
import warnings
from typing import List, Union
import cv2

def load_as_float(path):
    return imread(path).astype(np.float32)

#copied from TUM
def scale_intrinsics(
        intrinsics: Union[np.ndarray, torch.Tensor],
        h_ratio: Union[float, int],
        w_ratio: Union[float, int],
):
    r"""Scales the intrinsics appropriately for resized frames where
    :math:`h_\text{ratio} = h_\text{new} / h_\text{old}` and :math:`w_\text{ratio} = w_\text{new} / w_\text{old}`

    Args:
        intrinsics (numpy.ndarray or torch.Tensor): Intrinsics matrix of original frame
        h_ratio (float or int): Ratio of new frame's height to old frame's height
            :math:`h_\text{ratio} = h_\text{new} / h_\text{old}`
        w_ratio (float or int): Ratio of new frame's width to old frame's width
            :math:`w_\text{ratio} = w_\text{new} / w_\text{old}`

    Returns:
        numpy.ndarray or torch.Tensor: Intrinsics matrix scaled approprately for new frame size

    Shape:
        - intrinsics: :math:`(*, 3, 3)` or :math:`(*, 4, 4)`
        - Output: Matches `intrinsics` shape, :math:`(*, 3, 3)` or :math:`(*, 4, 4)`

    """
    if isinstance(intrinsics, np.ndarray):
        scaled_intrinsics = intrinsics.astype(np.float32).copy()
    elif torch.is_tensor(intrinsics):
        scaled_intrinsics = intrinsics.to(torch.float).clone()
    else:
        raise TypeError("Unsupported input intrinsics type {}".format(type(intrinsics)))
    if not (intrinsics.shape[-2:] == (3, 3) or intrinsics.shape[-2:] == (4, 4)):
        raise ValueError(
            "intrinsics must have shape (*, 3, 3) or (*, 4, 4), but had shape {} instead".format(
                intrinsics.shape
            )
        )
    if (intrinsics[..., -1, -1] != 1).any() or (intrinsics[..., 2, 2] != 1).any():
        warnings.warn(
            "Incorrect intrinsics: intrinsics[..., -1, -1] and intrinsics[..., 2, 2] should be 1."
        )

    scaled_intrinsics[..., 0, 0] *= w_ratio  # fx
    scaled_intrinsics[..., 1, 1] *= h_ratio  # fy
    scaled_intrinsics[..., 0, 2] *= w_ratio  # cx
    scaled_intrinsics[..., 1, 2] *= h_ratio  # cy
    return scaled_intrinsics

class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, skip_frames=1, dataset='kitti'):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.dataset = dataset
        self.k = skip_frames
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.jpg'))

            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length * self.k, len(imgs)-demi_length * self.k):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        height: int = 256  # 480/1.875, #leave default for sc-sfml compatibility
        width: int = 342
        cropped_width: int = 320
        height_downsample_ratio = float(height) / 480
        width_downsample_ratio = float(width) / 640
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        tgt_img2 = cv2.resize(tgt_img, (width, height), interpolation=cv2.INTER_LINEAR)
        margin = (tgt_img2.shape[1] - cropped_width)//2
        tgt_img2 = tgt_img2[:,margin:tgt_img2.shape[1]-margin,:] #adjusted size
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        ref_imgs2 = [] #adjusted size
        for i in ref_imgs[:]:
            temp_img = cv2.resize(i, (width, height), interpolation=cv2.INTER_LINEAR)
            margin = (temp_img.shape[1] - cropped_width) // 2
            ref_imgs2.append(temp_img[:, margin:temp_img.shape[1] - margin, :])
        if self.transform is not None:
            #imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics'])) original
            imgs, intrinsics = self.transform([tgt_img2] + ref_imgs2, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
            intrinsics = scale_intrinsics(intrinsics, height_downsample_ratio, width_downsample_ratio)
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)
