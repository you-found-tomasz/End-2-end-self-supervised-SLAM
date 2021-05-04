import torch

def gt_loss_wrapper(args, input_dict):
    gt = input_dict["depth"]
    predictions = input_dict["pred_depths"]
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