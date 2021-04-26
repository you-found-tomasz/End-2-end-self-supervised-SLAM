import os
import cv2
import numpy as np 
import torch
import imageio
from typing import Optional, Union

from torch.utils import data

from gradslam.geometry.geometryutils import relative_transformation
from gradslam.datasets import datautils

from tqdm import tqdm
import matplotlib.pyplot as plt

__all__ = ["NYU"]

class NYU(data.Dataset):

    def __init__(
        self,
        basedir: str,
        version: str = "rectified",
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
        return_depth: bool = True, 
        return_intrinsics: bool = True,  
    ):
        super(NYU, self).__init__()

        #Assignement 1
        basedir = os.path.join(basedir,version)

        self.height = height
        self.width = width
        self.height_downsample_ratio = float(height) / 253
        self.width_downsample_ratio = float(width) / 320
        self.channels_first = channels_first
        self.normalize_color = normalize_color
        self.return_depth = return_depth
        self.return_intrinsics = return_intrinsics
        
        ### SEQLEN, STRIDE, DILATION ###
        # Check if seqlen, stride and dilation are 'None' or int
        if not isinstance(seqlen, int):
            raise TypeError('"seqlen" must be int. Got {0}.'.format(type(seqlen)))
        if not (isinstance(stride, int) or stride is None):
            raise TypeError(
                '"stride" must be int or None. Got {0}.'.format(type(stride)))
        if not (isinstance(dilation, int) or dilation is None):
            raise TypeError(
                "dilation must be int or None. Got {0}.".format(type(dilation)))

        #Assignement 2
        dilation = dilation if dilation is not None else 0
        stride = stride if stride is not None else seqlen * (dilation + 1)
        self.seqlen = seqlen
        self.stride = stride
        self.dilation = dilation

        #Check if seqlen, stride and dilation values make sense
        if seqlen < 0:
            raise ValueError('"seqlen" must be positive. Got {0}.'.format(seqlen))
        if stride < 0:
            raise ValueError('"stride" must be positive. Got {0}.'.format(stride))
        if dilation < 0:
            raise ValueError('"dilation" must be positive. Got {0}.'.format(dilation))

        ### START, END ###
        # Check if start and end are 'None' or int
        if not (isinstance(start, int) or start is None):
            raise TypeError('"start" must be int or None. Got {0}.'.format(type(start)))
        if not (isinstance(end, int) or end is None):
            raise TypeError('"end" must be int or None. Got {0}.'.format(type(end)))

        #Assignement 3
        start = start if start is not None else 0
        self.start = start
        self.end = end

        #Check if start and end values make sense
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
        #sequences is a path to a textfile
        if isinstance(sequences, str):
            # read sequences from textfile
            if os.path.isfile(sequences):
                with open(sequences, "r") as f:
                    sequences = tuple(f.read().split("\n"))
            else:
                raise ValueError(
                    "incorrect filename: {} doesn't exist".format(sequences)
                )
        # sequences is already a touple or 'None'
        elif not (sequences is None or isinstance(sequences, tuple)):
            msg = '"sequences" should either be path to .txt file or tuple of sequence names or None, '
            msg += " but was of type {0} instead"
            raise TypeError(msg.format(type(sequences)))
        if isinstance(sequences, tuple):
            if len(sequences) == 0:
                raise ValueError(
                    '"sequences" must have atleast one element. Got len(sequences)=0'
                )

        sequence_paths = []
        # check folder structure for sequence:
        for item in os.listdir(basedir): # iterate over all items in basedir
            if os.path.isdir(os.path.join(basedir, item)): #is the item a folder?
                if sequences is None or (sequences is not None and item in sequences):
                    sequence_paths.append(os.path.join(basedir, item)) #add folder path of sequences

        # Check if folder paths list is not empty
        if len(sequence_paths) == 0:
            raise ValueError(
                'Incorrect folder structure in basedir ("{0}"). '.format(basedir)
            )
        
        #Check if folder path list is of equal length as provided sequence list
        if sequences is not None and len(sequence_paths) != len(sequences):
            msg = '"sequences" contains sequences not available in basedir:\n'
            msg += '"sequences" contains: ' + ", ".join(sequences) + "\n"
            msg += '"basedir" contains: ' + ", ".join(sequence_paths) + "\n"
            raise ValueError(msg)
        
        #Get a list of all color and depth files
        colorfiles, depthfiles = [], []
        idx = np.arange(seqlen) * (dilation + 1)
        for sequence_path in sequence_paths: #go through all sequences
            seq_colorfiles, seq_depthfiles = [], []
            for sequence_item in os.listdir(sequence_path): #go through all files in the folder
                if sequence_item.endswith(".jpg"):
                    seq_colorfiles.append(os.path.join(sequence_path,sequence_item)) #add to colorfile list if it is not a directory
            for sequence_item in os.listdir(os.path.join(sequence_path, "depth")): 
                seq_depthfiles.append(os.path.join(sequence_path,"depth",sequence_item)) # add to depthfile list
            
            # take start and stride into account for writing to colorfile and depthfile list
            num_frames = len(seq_colorfiles)
            for start_ind in range(0, num_frames, stride):
                if (start_ind + idx[-1]) >= num_frames:
                    break
                inds = start_ind + idx
                colorfiles.append([seq_colorfiles[i] for i in inds])
                depthfiles.append([seq_depthfiles[i] for i in inds])

        self.num_sequences = len(colorfiles)

        # Class members to store the list of valid filepaths.
        self.colorfiles = colorfiles
        self.depthfiles = depthfiles

        #Camera intrinsics
        intrinsics = torch.tensor(
            [[259.42895, 0, 162.79122,0], [0, 277.05046, 135.32596,0], [0, 0, 1,0], [0,0,0,1]]
        ).float()
        self.intrinsics = datautils.scale_intrinsics(
            intrinsics, self.height_downsample_ratio, self.width_downsample_ratio
        ).unsqueeze(0)

        # Scaling factor for depth images
        self.scaling_factor = 5000.0

    def __len__(self):
        r"""Returns the length of the dataset. """
        return self.num_sequences

    def __getitem__(self, idx: int):
        """Returns the data from the sequence at index idx.

        Returns:
            color_seq (torch.Tensor): Sequence of rgb images of each frame
            depth_seq (torch.Tensor): Sequence of depths of each frame                      
            intrinsics (torch.Tensor): Intrinsics for the current sequence
            framename (str): Name of the frame
            

        Shape:
            - color_seq: :math:`(L, H, W, 3)` if `channels_first` is False, else :math:`(L, 3, H, W)`. `L` denotes
                sequence length.
            - depth_seq: :math:`(L, H, W, 1)` if `channels_first` is False, else :math:`(L, 1, H, W)`. `L` denotes
                sequence length.
            - intrinsics: :math:`(1, 4, 4)`
        """

        # Read in the color, depth and intrinsics info
        color_seq_path = self.colorfiles[idx]
        depth_seq_path = self.depthfiles[idx]

        color_seq, depth_seq = [], []
        for i in range(self.seqlen):
            color = np.asarray(imageio.imread(color_seq_path[i]), dtype=float)
            color = self._preprocess_color(color)
            color = torch.from_numpy(color)
            color_seq.append(color)

            if self.return_depth:
                depth = np.asarray(imageio.imread(depth_seq_path[i]), dtype=np.int64)
                depth = self._preprocess_depth(depth)
                depth = torch.from_numpy(depth)
                depth_seq.append(depth)

        output = []
        color_seq = torch.stack(color_seq, 0).float()
        output.append(color_seq)

        if self.return_depth:
            depth_seq = torch.stack(depth_seq, 0).float()
            output.append(depth_seq)

        if self.return_intrinsics:
            intrinsics = self.intrinsics
            output.append(intrinsics)

        return tuple(output)

    
    def _preprocess_color(self, color: np.ndarray):
        """Preprocesses the color image by resizing to :math:`(H, W, C)`, (optionally) normalizing values to
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

    def _preprocess_depth(self, depth: np.ndarray):
        """Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        depth = cv2.resize(
            depth.astype(float),
            (self.width, self.height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth, -1)
        if self.channels_first:
            depth = datautils.channels_first(depth)
        return depth / self.scaling_factor