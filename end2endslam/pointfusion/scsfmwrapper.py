# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

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

#TODO: implement with args
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_FILE = "models/r18_rectified_nyu/dispnet_model_best.pth.tar"
PRETRAINED_DISPNET_PATH = os.path.join(CURR_DIR, MODEL_FILE)
RESNET_LAYERS = 18

class SCSfmWrapper(nn.Module):
    def __init__(self, device):
        super(SCSfmWrapper, self).__init__()
        self.device = device
        self.disp_net = DispResNet(RESNET_LAYERS, False)
        weights = torch.load(PRETRAINED_DISPNET_PATH, map_location=self.device)
        print("-> Loading model from ", PRETRAINED_DISPNET_PATH)
        #self.feed_height = loaded_dict_enc['height'] # Todo check if necessary
        #self.feed_width = loaded_dict_enc['width']
        self.disp_net.load_state_dict(weights['state_dict'])
        self.disp_net.to(self.device)
        self.disp_net.eval()


    def forward(self, images):
        # PREDICTION
        self.disp_net.train()
        
        input_image = images.to(self.device)
        outputs = self.disp_net(input_image)

        disp = outputs[0]

        #convert disparity map to absolute depths. Here the conversion
        #is hardcoded
        scaled_disp, depth = disp_to_depth(disp, 0.1, 100) #TODO: revise hard coded values
        return depth

    def get_depth_error(self, predictions, gt):
        
        #tensor_0 = torch.cuda.IntTensor(1).fill_(0)
        #tensor_1 = torch.cuda.IntTensor(1).fill_(1)
        #mask = torch.where(gt == 0, 0, 1)
        tensor_0 = torch.zeros(1, device = gt.device)
        tensor_1 = torch.ones(1, device = gt.device)
        mask = torch.where(gt == tensor_0, tensor_0, tensor_1)
        rmse = (gt - predictions) ** 2
        rmse = torch.sqrt(rmse.mean())

        rmse_log = (torch.log(gt) - torch.log(predictions)) ** 2
        rmse_log = torch.sqrt(rmse_log.mean())

        abs_rel = torch.mean(torch.abs(gt - predictions) / gt)
        abs = torch.mean(torch.abs(gt - predictions))

        sq_rel = torch.mean(((gt - predictions) ** 2) / gt)
        loss = {"abs": abs, "squ": sq_rel, "rmse": rmse, "rmse_log": rmse_log}
        return loss

    def pred_loss_depth(self, input_dict):
        images = input_dict["rgb"]
        gt = input_dict["depth"]
        predictions = self(images)
        error = self.get_depth_error(predictions, gt)
        return predictions, error["abs"]

    def pred_loss_reproj(self, input_dict):
        images = input_dict["rgb"]
        ref = input_dict["rgb_ref"]
        intrinsics = input_dict["intrinsic"]
        poses = input_dict["pose"]
        device = input_dict["device"]
        predictions = self(images)
        reprojected_images = image2image(images, ref, predictions, intrinsics, poses, device)
        error = self.get_reproj_error(reprojected_images, ref)
        return predictions, error

    def pred_loss_depth_consistency(self, input_dict, slam, pointclouds, prev_frame, DEBUG_PATH):
        # get inputs
        colors = input_dict["rgb"]
        intrinsics = input_dict["intrinsic"]
        poses = input_dict["pose"]
    
        # predict depth
        pred_depths = self(colors)

        # Ground truth Depth for comparison (only for debug)
        if DEBUG_PATH:
            gt_depths = input_dict["depth"]
            gt_depths = torch.nn.functional.interpolate(input=gt_depths, size=(120, 160), mode="nearest")
            gt_depths_u = torch.unsqueeze(gt_depths, 1).permute(0, 1, 3, 4, 2)

        # Downsample (since depth prediction does not work in (120,160))
        colors = torch.nn.functional.interpolate(input=colors, size=(120, 160), mode="bicubic")
        pred_depths = torch.nn.functional.interpolate(input=pred_depths, size=(120, 160), mode="nearest")

        # added artificial sequence length dimension and then don't use it (stupid but necessary)
        # permute since slam does NOT support channels_first = True
        colors_u = torch.unsqueeze(colors, 1).permute(0, 1, 3, 4, 2)
        pred_depths_u = torch.unsqueeze(pred_depths, 1).permute(0, 1, 3, 4, 2)
        poses_u = torch.unsqueeze(poses, 1)

        # SLAM
        rgbdimages = RGBDImages(colors_u, pred_depths_u, intrinsics, poses_u, channels_first=False, device=self.device)
        live_frame = rgbdimages[:, 0]
        pointclouds, live_frame.poses = slam.step(pointclouds, live_frame, prev_frame)

        # Get depth map from SLAM # TODO: can try again with find_correspondences to incorporate uncertainty?
        proj_colors, proj_depths = depthfromslam(pointclouds, live_frame)

        # Compute Loss
        error = self.get_depth_error(pred_depths, proj_depths[:, :, :, :, 0].detach())

        # Visualizations: Change path
        if DEBUG_PATH:
            # Projected Color Vis
            vis_proj_color = proj_colors[0, 0, :, :, :].detach().cpu().numpy()
            imageio.imwrite(os.path.join(DEBUG_PATH, "debug_color_proj.png"), vis_proj_color)
            # Projected Depth Vis
            vis_proj_depth = proj_depths[0, 0, :, :, :].detach().cpu().numpy()
            vmax = np.percentile(vis_proj_depth, 95)
            vmin = vis_proj_depth.min()
            normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            proj_depth_np = (mapper.to_rgba(vis_proj_depth[:, :, 0])[:, :, :3] * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(DEBUG_PATH, "debug_depth_proj.png"), proj_depth_np)
            # Color vis
            vis_color = live_frame.rgb_image[0, 0, :, :, :].detach().cpu().numpy()
            imageio.imwrite(os.path.join(DEBUG_PATH, "debug_color.png"), vis_color)
            # Depth Vis
            vis_pred_depth = live_frame.depth_image[0, 0, :, :, :].detach().cpu().numpy()
            vmax = np.percentile(vis_pred_depth, 95)
            vmin = vis_pred_depth.min()
            normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            pred_depth_np = (mapper.to_rgba(vis_pred_depth[:, :, 0])[:, :, :3] * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(DEBUG_PATH, "debug_depth_proj.png"), pred_depth_np)
            # SLAM Vis
            import open3d as o3d
            o3d.visualization.draw_geometries([pointclouds.open3d(0)])

        return pred_depths, error["abs"], slam, pointclouds, live_frame

    def get_reproj_error(self, pred, ref):
        abs = torch.abs(pred - ref).mean([-1, -2, -3])
        return torch.mean(abs)


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def depthfromslam(pointclouds, live_frame):
    # Similar to gradslam.fusionutils.find_active_map_points
    # Todo: take confidence into account

    batch_size, seq_len, height, width = live_frame.shape

    # Transform pointcloud to live frame pose
    tinv = inverse_transformation(live_frame.poses.squeeze(1))
    pointclouds_transformed = pointclouds.transform(tinv)
    # don't consider missing depth values (z < 0)
    is_front_of_plane = (
        pointclouds_transformed.points_padded[..., -1] > 0
    )

    # Get depths from transformed pointcloud
    depths_padded = pointclouds_transformed.points_padded[..., -1]
    depths_padded = torch.unsqueeze(depths_padded, 2)

    # Project pointcloud into live frame (IN PLACE operation)
    pointclouds_transformed.pinhole_projection_(
        live_frame.intrinsics.squeeze(1)
    )

    # Discard depth, only keep width and height
    img_plane_points = pointclouds_transformed.points_padded[..., :-1]
    # Mask for eliminating points that are not inside image
    is_in_frame = (
        (img_plane_points[..., 0] > -1e-3)
        & (img_plane_points[..., 0] < width - 0.999)
        & (img_plane_points[..., 1] > -1e-3)
        & (img_plane_points[..., 1] < height - 0.999)
        & is_front_of_plane
        & pointclouds.nonpad_mask
    )
    in_plane_pos = img_plane_points.round().long()
    in_plane_pos = torch.cat(
        [
            in_plane_pos[..., 1:2].clamp(0, height - 1),
            in_plane_pos[..., 0:1].clamp(0, width - 1),
        ],
        -1,
    )  # height, width
    batch_size, num_points = in_plane_pos.shape[:2]
    batch_point_idx = (
        create_meshgrid(batch_size, num_points, normalized_coords=False)
        .squeeze(0)
        .to(pointclouds.device)
    )
    # Create a lookup table pc2im_bnhw that stores all points from pointcloud (referenced by point_cloud_index,
    # that are active in current frame (at pixel height_index, width_index).
    # Shape: (num_points, 4) with 4 columns [batch_index, point_cloud_index, height_index, width_index]
    idx_and_plane_pos = torch.cat([batch_point_idx.long(), in_plane_pos], -1)
    pc2im_bnhw = idx_and_plane_pos[is_in_frame]  # (?, 4)

    # Use lookup table to get depth of active points (= reprojected depth)
    proj_depth = torch.zeros_like(live_frame.depth_image)
    proj_depth[pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3], :] = \
        depths_padded[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]]

    # Use lookup table to get color of active points (= reprojected color)
    proj_colors = torch.zeros_like(live_frame.rgb_image)
    proj_colors[pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3], :] = \
        pointclouds.colors_padded[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]]

    if pc2im_bnhw.shape[0] == 0:
        warnings.warn("No active map points were found")

    return proj_colors, proj_depth


if __name__ == '__main__':
    print("Don't call wrapper directly for now!")

