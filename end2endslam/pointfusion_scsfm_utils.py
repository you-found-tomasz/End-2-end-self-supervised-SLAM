import torch
import numpy as np
from gradslam.structures.rgbdimages import RGBDImages


def compute_scaling_coef(args, input_dict):
    if args.dataset == 'tum':
        stacked_pred_depth = np.vstack(input_dict["pred_depths"][0].detach().cpu().squeeze().numpy())
        # Ignoring zero (unknown) depth values in gt
        stacked_gt_depth = np.vstack(input_dict["depth"].detach().cpu().squeeze().numpy())
        stacked_gt_depth[stacked_gt_depth == 0] = np.nan

        gt_min = np.nanmin(stacked_gt_depth)
        gt_max = np.nanmax(stacked_gt_depth)
        gt_mean = np.nanmean(stacked_gt_depth)
        gt_median = np.nanmedian(stacked_gt_depth)
        gt_std = np.nanstd(stacked_gt_depth)
        pred_min = np.min(stacked_pred_depth)
        pred_max = np.max(stacked_pred_depth)
        pred_mean = np.mean(stacked_pred_depth)
        pred_median = np.median(stacked_pred_depth)
        pred_std = np.std(stacked_pred_depth)

        scaling_coeff = gt_median / pred_median
        print("Scaling coefficient: {}".format(scaling_coeff))
        print("Mean (gt, pred): {}, {}".format(gt_mean, pred_mean))
        print("Median (gt, pred): {}, {}".format(gt_median, pred_median))
        print("Min (gt, pred): {}, {}".format(gt_min, pred_min))
        print("Max (gt, pred): {}, {}".format(gt_max, pred_max))
        print("Std (gt, pred): {}, {}".format(gt_std, pred_std))

    else:
        scaling_coeff = 1

    return scaling_coeff, pred_min*scaling_coeff, pred_max*scaling_coeff

def slam_step(input_dict, slam, pointclouds, prev_frame, device, args):
    """ Perform SLAM step
    """
    # get inputs
    intrinsics = input_dict["intrinsic_slam"]
    colors = input_dict["rgb_slam"]
    # pass identity as poses (important for first frame, dummy for rest)
    if args.odometry == "gt":
        poses = input_dict["gt_poses"]
    else:
        poses = torch.eye(4, device=device).view(1, 4, 4).repeat(colors.shape[0], 1, 1)
    pred_depths = input_dict["pred_depths_slam"]

    # added artificial sequence length dimension and then don't use it (stupid but necessary)
    # permute since slam does NOT support channels_first = True
    colors_u = torch.unsqueeze(colors, 1).permute(0, 1, 3, 4, 2)
    pred_depths_u = torch.unsqueeze(pred_depths, 1).permute(0, 1, 3, 4, 2)
    poses_u = torch.unsqueeze(poses, 1)

    # SLAM
    rgbdimages = RGBDImages(colors_u, pred_depths_u, intrinsics, poses_u, channels_first=False, device=device)
    live_frame = rgbdimages[:, 0]
    pointclouds, live_frame.poses = slam.step(pointclouds, live_frame, prev_frame)

    # Compute relative poses for reprojection
    if prev_frame == None:
        relative_pose = live_frame.poses
    else:
        relative_pose = torch.matmul(torch.inverse(prev_frame.poses), live_frame.poses)

    return slam, pointclouds, live_frame, relative_pose

def compute_relative_pose_magnitudes(rel_pose):
    # rel_pose: Bx4x4 matrix
    batch_size = rel_pose.shape[0]
    mag_transl = 0
    mag_rot = 0
    for i in range(batch_size):
        # translation
        transl = rel_pose[i, 0:3, 3]
        mag_transl += sum(transl*transl)**0.5
        # rotation (via angle axis)
        rot_mat = rel_pose[i, 0:3,0:3]
        mag_rot += np.arccos((rot_mat[0,0] + rot_mat[1,1] + rot_mat[2,2] -1 )/2)
    mag_rot = mag_rot / batch_size
    mag_transl = mag_transl / batch_size
    return mag_transl, mag_rot

