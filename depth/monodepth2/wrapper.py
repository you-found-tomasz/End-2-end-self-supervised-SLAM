import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

from .networks import ResnetEncoder, DepthDecoder
from .layers import disp_to_depth
from .utils import download_model_if_doesnt_exist
from .evaluate_depth import compute_errors

class wrapper(object): 
    def __init__(self, device="cuda", train=False, model_name=None):
        self.model_name = "mono_640x192" if model_name is None else model_name
        self.device = device
        if not device in ["cuda", "cpu"]:
            raise ValueError("Wrong device specified {}".format(self.device))

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        download_model_if_doesnt_exist(self.model_name)

    def init_model(self):
        model_path = os.path.join("models", self.model_name)
        print("-> Loading model from ", model_path)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        # LOADING PRETRAINED MODEL
        self.encoder = ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=self.device)

        # extract the height and width of image that this model was trained with
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()

        self.depth_decoder = DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(depth_decoder_path, map_location=self.device)
        self.depth_decoder.load_state_dict(loaded_dict)

        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()
        print("   Successfully Loaded pretrained encoder and decoder   ")

    
    def predict_depth(self, image):
        """Predict depth map through Monodepth model using prepped image

        Args:
            data ([tensor_image]): [RGB Image without unsqueeze]
        """
        if not isinstance(image, torch.Tensor):
            image = torch.Tensor(image, requires_grad=False)
        image = image.to(self.device)
        orig_height, orig_width = image.size(-3), image.size(-2)

        image = torch.nn.functional.interpolate(
                image, (self.height, self.width), mode="bilinear", align_corners=False)
        features = self.encoder(image)
        outputs = self.depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp = disp_to_depth(disp)
        disp_resized = torch.nn.functional.interpolate(disp, (orig_width, orig_height), mode="bilinear", align_corners=False)

        return disp, disp_resized

    def get_error(self, gt_depth, pred):
        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_errors(gt_depth, pred)
        error = dict()
        error["abs"] = abs_rel
        error["squ"] = sq_rel
        error["rmse"] = rmse
        error["rmse_log"] = rmse_log
        return error

    def prepare_input(self, image):
        # Check if some scaling etc is needed
        return image

    def predict(self, image):
        