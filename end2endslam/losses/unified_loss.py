from __future__ import absolute_import, division, print_function

from end2endslam.losses.photo_and_geometry_loss import photo_and_geometry_loss_wrapper
from end2endslam.losses.smooth_loss import smooth_loss_wrapper
from end2endslam.losses.gt_loss import gt_loss_wrapper
from end2endslam.losses.depth_consistency_loss import depth_consistency_loss_wrapper


def pred_loss_unified(args, input_dict, slam, pointclouds, live_frame):
    loss = {"com": 0}

    if args.loss_photo_factor > 0 or args.loss_geom_factor > 0:
        loss["photo"], loss["geom"] = photo_and_geometry_loss_wrapper(args, input_dict)
        loss["com"] += loss["photo"] * args.loss_photo_factor + loss["geom"] * args.loss_geom_factor
    if args.loss_smooth_factor > 0:
        loss["smooth"] = smooth_loss_wrapper(args, input_dict)
        loss["com"] += loss["smooth"] * args.loss_smooth_factor
    if args.loss_cons_factor > 0:
        loss["cons"] = depth_consistency_loss_wrapper(args, input_dict, slam, pointclouds, live_frame)
        loss["com"] += loss["cons"] * args.loss_cons_factor
    if args.loss_gt_factor > 0:
        loss["gt"] = gt_loss_wrapper(args, input_dict)
        loss["com"] += loss["gt"] * args.loss_gt_factor

    if args.loss_photo_factor + args.loss_geom_factor + args.loss_smooth_factor + args.loss_cons_factor + args.loss_gt_factor <= 0:
        raise ValueError("The loss multiplication factors are less than zero | Not training")

    return loss