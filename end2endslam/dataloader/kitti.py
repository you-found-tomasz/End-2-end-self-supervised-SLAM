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
    r"""A torch Dataset for loading in `the KITTI odometry dataset.
    Will fetch sequences of rgb images, depth maps, intrinsics matrix, poses, frame to frame relative transformations
    (with first frame's pose as the reference transformation), names of frames. Uses extracted `.tgz` sequences
    downloaded from `here <https://vision.in.tum.de/data/datasets/rgbd-dataset/download>`__.
    Expects similar to the following folder structure for the TUM dataset:

    .. code-block::


        | ├── TUM
        | │   ├── rgbd_dataset_freiburg1_rpy
        | │   │   ├── depth/
        | │   │   ├── rgb/
        | │   │   ├── accelerometer.txt
        | │   │   ├── depth.txt
        | │   │   ├── groundtruth.txt
        | │   │   └── rgb.txt
        | │   ├── rgbd_dataset_freiburg1_xyz
        | │   │   ├── depth/
        | │   │   ├── rgb/
        | │   │   ├── accelerometer.txt
        | │   │   ├── depth.txt
        | │   │   ├── groundtruth.txt
        | │   │   └── rgb.txt
        | │   ├── ...
        |
        |

    Example of sequence creation from frames with `seqlen=4`, `dilation=1`, `stride=3`, and `start=2`:

    .. code-block::


                                            sequence0
                        ┎───────────────┲───────────────┲───────────────┒
                        |               |               |               |
        frame0  frame1  frame2  frame3  frame4  frame5  frame6  frame7  frame8  frame9  frame10  frame11 ...
                                                |               |               |                |
                                                └───────────────┵───────────────┵────────────────┚
                                                                    sequence1

    Args:
        basedir (str): Path to the base directory containing extracted TUM sequences in separate directories.
            Each sequence subdirectory is assumed to contain `depth/`, `rgb/`, `accelerometer.txt`, `depth.txt` and
            `groundtruth.txt` and `rgb.txt`, E.g.:

            .. code-block::


                ├── rgbd_dataset_freiburgX_NAME
                │   ├── depth/
                │   ├── rgb/
                │   ├── accelerometer.txt
                │   ├── depth.txt
                │   ├── groundtruth.txt
                │   └── rgb.txt

        sequences (str or tuple of str or None): Sequences to use from those available in `basedir`.
            Can be path to a `.txt` file where each line is a sequence name (e.g. `rgbd_dataset_freiburg1_rpy`),
            a tuple of sequence names, or None to use all sequences. Default: None
        seqlen (int): Number of frames to use for each sequence of frames. Default: 4
        dilation (int or None): Number of (original trajectory's) frames to skip between two consecutive
            frames in the extracted sequence. See above example if unsure.
            If None, will set `dilation = 0`. Default: None
        stride (int or None): Number of frames between the first frames of two consecutive extracted sequences.
            See above example if unsure. If None, will set `stride = seqlen * (dilation + 1)`
            (non-overlapping sequences). Default: None
        start (int or None): Index of the rgb frame from which to start extracting sequences for every sequence.
            If None, will start from the first frame. Default: None
        end (int): Index of the rgb frame at which to stop extracting sequences for every sequence.
            If None, will continue extracting frames until the end of the sequence. Default: None
        height (int): Spatial height to resize frames to. Default: 480
        width (int): Spatial width to resize frames to. Default: 640
        channels_first (bool): If True, will use channels first representation :math:`(B, L, C, H, W)` for images
            `(batchsize, sequencelength, channels, height, width)`. If False, will use channels last representation
            :math:`(B, L, H, W, C)`. Default: False
        normalize_color (bool): Normalize color to range :math:`[0 1]` or leave it at range :math:`[0 255]`.
            Default: False
        return_depth (bool): Determines whether to return depths. Default: True
        return_intrinsics (bool): Determines whether to return intrinsics. Default: True
        return_pose (bool): Determines whether to return poses. Default: True
        return_transform (bool): Determines whether to return transforms w.r.t. initial pose being transformed to be
            identity. Default: True
        return_names (bool): Determines whether to return sequence names. Default: True
        return_timestamps (bool): Determines whether to return rgb, depth and pose timestamps. Default: True


    Examples::

        >>> dataset = TUM(
            basedir="TUM-data/",
            sequences=("rgbd_dataset_freiburg1_rpy", "rgbd_dataset_freiburg1_xyz"))
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
        height: int = 480,
        width: int = 640,
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
        self.height_downsample_ratio = float(height) / 480
        self.width_downsample_ratio = float(width) / 640
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

        # check if TUM folder structure correct: If sequences is not None, should contain all sequence paths.
        # Should also contain atleast one sequence path.
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

        # Camera intrinsics matrix for KITTI dataset #TODO
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


