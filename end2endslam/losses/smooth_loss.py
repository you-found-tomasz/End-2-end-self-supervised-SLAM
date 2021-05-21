from end2endslam.losses.loss_utils.scsfm_loss import compute_smooth_loss

def smooth_loss_wrapper(args, input_dict):
    tgt_img = input_dict["rgb"]
    ref_imgs = [input_dict["rgb_ref"]]

    # multi-scale
    tgt_depth = input_dict["pred_depths"]
    ref_depths = [input_dict["pred_depths_ref"]]

    smooth_loss = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

    return smooth_loss