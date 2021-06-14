import torch
from kornia.geometry.linalg import inverse_transformation
from gradslam.geometry.geometryutils import create_meshgrid

def depth_consistency_loss_wrapper(args, input_dict, slam, pointclouds, live_frame):
    cons_loss = pred_loss_depth_consistency(input_dict, slam, pointclouds, live_frame)
    return cons_loss


def pred_loss_depth_consistency(input_dict, slam, pointclouds, live_frame, DEBUG_PATH=""):
    # get inputs
    colors = input_dict["rgb"]
    colors_slam = input_dict["rgb_slam"]  # downscaled
    intrinsics = input_dict["intrinsic_slam"]
    poses = input_dict["pose"]
    pred_depths = input_dict["pred_depths"]
    pred_depths_slam = input_dict["pred_depths_slam"]  # downscaled

    # Ground truth Depth for comparison (only for debug)
    if DEBUG_PATH:
        gt_depths = input_dict["depth"]
        gt_depths = torch.nn.functional.interpolate(input=gt_depths, size=(120, 160), mode="nearest")
        gt_depths_u = torch.unsqueeze(gt_depths, 1).permute(0, 1, 3, 4, 2)

    # imporant, the loss is compute in slam resolution (downscaled)

    # Get depth map from SLAM # TODO: can try again with find_correspondences to incorporate uncertainty?
    proj_colors, proj_depths = depthfromslam(pointclouds, live_frame)

    # Compute Loss
    error = get_depth_error(pred_depths_slam, proj_depths[:, :, :, :, 0].detach())

    return error["abs"]

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
        print("No active map points were found")

    return proj_colors, proj_depth

def get_depth_error(predictions, gt):
    # tensor_0 = torch.cuda.IntTensor(1).fill_(0)
    # tensor_1 = torch.cuda.IntTensor(1).fill_(1)
    # mask = torch.where(gt == 0, 0, 1)
    tensor_0 = torch.zeros(1, device=gt.device)
    tensor_1 = torch.ones(1, device=gt.device)
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


