import torch

def gt_loss_wrapper(args, input_dict):
    """ Compute loss w.r.t gt works only at full scale
    """
    gt = input_dict["depth"]
    predictions = input_dict["pred_depths"]
    predictions = predictions[0]
    gt_loss = get_depth_error(predictions, gt)
    return gt_loss["abs"]

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



@torch.no_grad()
def compute_errors(gt, pred, dataset='tum'):
    """ Compute gt error, from scsfml learner

    error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']

    Args:
        gt: list of gt images
        pred. list of pred images (same size as gt)

    """
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0
    batch_size, h, w = gt.size()

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if dataset == 'kitti':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 80

    if dataset == 'nyu' or dataset == 'tum':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.09375 * gt.size(1)), int(0.98125 * gt.size(1))
        x1, x2 = int(0.0640625 * gt.size(2)), int(0.9390625 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 10

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0.1) & (current_gt < max_depth)
        valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, max_depth)

        #rescales prediction to match gt scale
        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]