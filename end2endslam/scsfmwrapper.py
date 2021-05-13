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

        #Create depth prediction network and load pretrained weights
        self.disp_net = DispResNet(resnet_layers, False)#.to(device)
        if pretrained:
            weights = torch.load(pretrained_path, map_location=self.device)
            print("-> Loading model from ", pretrained_path)
            #self.feed_height = loaded_dict_enc['height'] # Todo check if necessary
            #self.feed_width = loaded_dict_enc['width']
            self.disp_net.load_state_dict(weights['state_dict'])
        
        self.disp_net.to(self.device)

        #self.disp_net.eval() #set in eval mode... this CHANGES the behaviour of the net
        self.disp_net.train() #set in train mode... this CHANGES the behaviour of the net

    def forward(self, images):
        #images is in format: batch x color_channels x height x width

        #input_images = images.to(self.device) #PERFORMANCE!

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
            single_depth = 1 / single_disp
            depth_list.append(single_depth)

        # Old: Single scale
        #disp = outputs[0] #take biggest scale
        #depth = 1 / disp  # compute depth from disparity
        #return depth

        return depth_list

    



if __name__ == '__main__':
    print("Don't call wrapper directly for now!")

