import os
from typing import Optional, Union

import cv2
import imageio
import numpy as np
import torch
from end2endslam.dataloader.dataloader_utils.pykitti_odometry import odometry
from gradslam.geometry.geometryutils import relative_transformation
from torch.utils import data

from gradslam.datasets import datautils

__all__ = ["KITTI"]


class KITTI(data.Dataset):
    r"""
    Examples::

        >>> dataset = KITTI(
            basedir="KITTI-data/",
            sequences="sequences.txt"
        >>> loader = data.DataLoader(dataset=dataset, batch_size=4)
        >>> colors, depths, intrinsics, poses, transforms, names = next(iter(loader))

    """

    def __init__(
        self,
        basedir: str,
        sequences: Union[tuple, str, None] = None,
        seqlen: int = 4,
        dilation: Optional[int] = None,
        stride: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        height: int = 512,
        width: int = 1392,
        channels_first: bool = False,
        normalize_color: bool = False,
        *,
        return_dummy_depth: bool = True,
        return_intrinsics: bool = True,
        return_pose: bool = True,
    ):
        super(KITTI, self).__init__()

        basedir = os.path.normpath(basedir)
        self.height = height
        self.width = width
        self.height_downsample_ratio = float(height) / 512
        self.width_downsample_ratio = float(width) / 1392
        self.channels_first = channels_first
        self.normalize_color = normalize_color
        self.return_intrinsics = return_intrinsics
        self.return_pose = return_pose
        self.return_dummy_depth = return_dummy_depth

        if not isinstance(seqlen, int):
            raise TypeError('"seqlen" must be int. Got {0}.'.format(type(seqlen)))
        if not (isinstance(stride, int) or stride is None):
            raise TypeError(
                '"stride" must be int or None. Got {0}.'.format(type(stride))
            )
        if not (isinstance(dilation, int) or dilation is None):
            raise TypeError(
                "dilation must be int or None. Got {0}.".format(type(dilation))
            )
        dilation = dilation if dilation is not None else 0
        stride = stride if stride is not None else seqlen * (dilation + 1)
        self.seqlen = seqlen
        self.stride = stride
        self.dilation = dilation
        if seqlen < 0:
            raise ValueError('"seqlen" must be positive. Got {0}.'.format(seqlen))
        if dilation < 0:
            raise ValueError('"dilation" must be positive. Got {0}.'.format(dilation))
        if stride < 0:
            raise ValueError('"stride" must be positive. Got {0}.'.format(stride))

        if not (isinstance(start, int) or start is None):
            raise TypeError('"start" must be int or None. Got {0}.'.format(type(start)))
        if not (isinstance(end, int) or end is None):
            raise TypeError('"end" must be int or None. Got {0}.'.format(type(end)))
        start = start if start is not None else 0
        self.start = start
        self.end = end
        if start is not None and start < 0:
            raise ValueError(
                '"start" must be None or positive. Got {0}.'.format(stride)
            )
        if not (end is None or end > start):
            raise ValueError(
                '"end" ({0}) must be None or greater than start ({1})'.format(
                    end, start
                )
            )

        # preprocess sequences to be a tuple or None
        if isinstance(sequences, str):
            if os.path.isfile(sequences):
                with open(sequences, "r") as f:
                    sequences = tuple(f.read().split("\n"))[:-1]
            else:
                raise ValueError(
                    "incorrect filename: {} doesn't exist".format(sequences)
                )
        else:
            raise ValueError(
                "Sequences must be .txt file with sequences"
            )

        sequence_paths = []
        # check folder structure for sequence:
        for item in os.listdir(os.path.join(basedir, "sequences")):
            if os.path.isdir(os.path.join(basedir, "sequences", item)):
                if sequences is None or (sequences is not None and item in sequences):
                    sequence_paths.append(os.path.join(basedir, "sequences", item))
        if len(sequence_paths) == 0:
            raise ValueError(
                'Incorrect folder structure in basedir ("{0}"). '.format(basedir)
            )
        if sequences is not None and len(sequence_paths) != len(sequences):
            msg = '"sequences" contains sequences not available in basedir:\n'
            msg += '"sequences" contains: ' + ", ".join(sequences) + "\n"
            msg += '"basedir" contains: ' + ", ".join(sequence_paths) + "\n"
            raise ValueError(msg)

        # get association and pose file paths
        colorfiles, depthfiles, poses = [], [], []
        idx = np.arange(seqlen) * (dilation + 1)
        for sequence_path in sequence_paths:
            seq_name = os.path.split(sequence_path)[-1]
            seq_poses = odometry(basedir, seq_name).poses
            seq_colorfiles = []
            for filepath in sorted(os.listdir(os.path.join(sequence_path, "image_3"))):
                seq_colorfiles.append(os.path.join(sequence_path, "image_3", filepath)) #Todo: Remove hard-coded image_3
            num_frames = len(seq_colorfiles)
            for start_ind in range(0, num_frames, stride):
                if (start_ind + idx[-1]) >= num_frames:
                    break
                inds = start_ind + idx
                colorfiles.append([seq_colorfiles[i] for i in inds])

                if self.return_pose:
                    poses.append([seq_poses[i] for i in inds])

        self.num_sequences = len(colorfiles)

        # Class members to store the list of valid filepaths.
        self.colorfiles = colorfiles
        self.poses = poses

        # Camera intrinsics matrix for KITTI dataset #TODO: not only cam3
        K_cam3 = odometry(basedir, seq_name).calib.K_cam3
        intrinsics_np = np.zeros((4, 4))
        intrinsics_np[:3, :3] = K_cam3
        intrinsics_np[3, 3] = 1
        intrinsics = torch.tensor(
            intrinsics_np
        ).float()
        self.intrinsics = datautils.scale_intrinsics(
            intrinsics, self.height_downsample_ratio, self.width_downsample_ratio
        ).unsqueeze(0)

    def __len__(self):
        r"""Returns the length of the dataset. """
        return self.num_sequences

    def __getitem__(self, idx: int):
        r"""Returns the data from the sequence at index idx.

        Returns:
            color_seq (torch.Tensor): Sequence of rgb images of each frame
            depth_seq (torch.Tensor): Sequence of depths of each frame
            pose_seq (torch.Tensor): Sequence of poses of each frame
            transform_seq (torch.Tensor): Sequence of transformations between each frame in the sequence and the
                previous frame. Transformations are w.r.t. the first frame in the sequence having identity pose
                (relative transformations with first frame's pose as the reference transformation). First
                transformation in the sequence will always be `torch.eye(4)`.
            intrinsics (torch.Tensor): Intrinsics for the current sequence
            framename (str): Name of the frame
            timestamp_seq (str): Sequence of timestamps of matched rgb, depth and pose stored
                as "rgb rgb_timestamp depth depth_timestamp pose pose_timestamp\n".

        Shape:
            - color_seq: :math:`(L, H, W, 3)` if `channels_first` is False, else :math:`(L, 3, H, W)`. `L` denotes
                sequence length.
            - depth_seq: :math:`(L, H, W, 1)` if `channels_first` is False, else :math:`(L, 1, H, W)`. `L` denotes
                sequence length.
            - pose_seq: :math:`(L, 4, 4)` where `L` denotes sequence length.
            - transform_seq: :math:`(L, 4, 4)` where `L` denotes sequence length.
            - intrinsics: :math:`(1, 4, 4)`
        """

        # Read in the color, depth, pose, label and intrinstics info.
        color_seq_path = self.colorfiles[idx]

        color_seq, pose_seq = [], []
        for i in range(self.seqlen):
            color = np.asarray(imageio.imread(color_seq_path[i]), dtype=float)
            color = self._preprocess_color(color)
            color = torch.from_numpy(color)
            color_seq.append(color)

        output = []
        color_seq = torch.stack(color_seq, 0).float()
        output.append(color_seq)

        if self.return_dummy_depth:
            output.append([])

        if self.return_intrinsics:
            intrinsics = self.intrinsics
            output.append(intrinsics)

        if self.return_pose:
            poses = self.poses[idx]
            pose_seq = [torch.from_numpy(pose) for pose in poses]
            pose_seq = torch.stack(pose_seq, 0).float()
            pose_seq = self._preprocess_poses(pose_seq)
            output.append(pose_seq)

        return tuple(output)

    def _preprocess_color(self, color: np.ndarray):
        r"""Preprocesses the color image by resizing to :math:`(H, W, C)`, (optionally) normalizing values to
        :math:`[0, 1]`, and (optionally) using channels first :math:`(C, H, W)` representation.

        Args:
            color (np.ndarray): Raw input rgb image

        Retruns:
            np.ndarray: Preprocessed rgb image

        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
        """
        color = cv2.resize(
            color, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        if self.normalize_color:
            color = datautils.normalize_image(color)
        if self.channels_first:
            color = datautils.channels_first(color)
        return color

    def _preprocess_poses(self, poses: torch.Tensor):
        r"""Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
        homogenous transformation for all other poses.

        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed

        Returns:
            Output (torch.Tensor): Preprocessed poses

        Shape:
            - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
            - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        """
        return relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1), poses
        )


