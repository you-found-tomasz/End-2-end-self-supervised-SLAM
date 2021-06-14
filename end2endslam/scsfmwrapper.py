#Wrapper for the sc-sfml depth prediction network

from __future__ import absolute_import, division, print_function

import warnings
import torch.nn as nn
import os
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
#from perception.SC_SfMLearner_Release.models import DispResNet

from models import DispResNet

import torch
from gradslam.structures.rgbdimages import RGBDImages
import imageio
#import open3d as o3d
from kornia.geometry.linalg import inverse_transformation
from gradslam.geometry.geometryutils import create_meshgrid 
#from end2endslam.loss_hamza.reprojection_loss import image2image
from loss_hamza.reprojection_loss import image2image


class SCSfmWrapper(nn.Module):
    def __init__(self, device,pretrained=True,pretrained_path=None,resnet_layers=18):
        """
        Args:
            device: cuda or cpu
            pretrained: whether to load pretrained weights or not
            pretrained_path: path to pretrained model, only used if pretrained=True
            resnet_layers: number of resnet layers: 18 or 54
        """
        super(SCSfmWrapper, self).__init__()
        self.device = device
        self.scale_coeff = 1

        #Create depth prediction network and load pretrained weights
        self.disp_net = DispResNet(resnet_layers, False)#.to(device)
        if pretrained:
            weights = torch.load(pretrained_path, map_location=self.device)
            print("-> Loading model from ", pretrained_path)
            self.disp_net.load_state_dict(weights['state_dict'])
        
        self.disp_net.to(self.device)
        self.disp_net.train()

    def forward(self, images):
        #images are in format: batch x color_channels x height x width
        #apply hardcoded normalization (as done by scsfml)
        #input images have value in range [0,1]

        input_images = ((images - 0.45)/0.225).to(self.device)
        outputs = self.disp_net(input_images)

        # Multi scale
        disp = outputs # multiscale
        if not self.disp_net.training:
            disp = [disp]
        depth_list = []
        for single_disp in disp:
            single_depth = 1 / single_disp * self.scale_coeff
            depth_list.append(single_depth)

        return depth_list

    def save_model(self, save_path, epoch, loss):
        torch.save({
            'epoch': epoch,
            'state_dict': self.disp_net.state_dict(),
            'loss': loss
        }, save_path)



if __name__ == '__main__':
    print("Don't call wrapper directly!")

