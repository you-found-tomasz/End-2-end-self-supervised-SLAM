import torch
from end2endslam.losses.loss_utils.scsfm_loss import compute_photo_and_geometry_loss

def photo_and_geometry_loss_wrapper(args, input_dict):
    # certain stuff needs to be wrapped in lists
    intrinsics = input_dict["intrinsic"]
    intrinsics = intrinsics[:, 0, 0:3, 0:3]
    poses = input_dict["pose"]
    poses = [poses[:, 0, 0:3, :]]
    tgt_img = input_dict["rgb"]
    ref_imgs = [input_dict["rgb_ref"]]
    tgt_depth = [input_dict["pred_depths"]]
    ref_depths = [[input_dict["pred_depths_ref"]]]
    poses_inv = poses # Dummy, NOT used inside function call!
    photo_loss, geom_loss = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                    poses, poses_inv, max_scales=1, with_ssim=1,
                                    with_mask=1, with_auto_mask=0, padding_mode='zeros')
    return photo_loss, geom_loss
