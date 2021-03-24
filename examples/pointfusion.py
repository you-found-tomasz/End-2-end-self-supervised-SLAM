from argparse import ArgumentParser, RawTextHelpFormatter

import open3d as o3d
import torch
from torch.utils.data import DataLoader

from gradslam.datasets.icl import ICL
from gradslam.datasets.tum import TUM
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.rgbdimages import RGBDImages
from loss_hamza.reprojection_loss import get_indexed_projection_TUM, image2image
from depth.monodepth2.monodepthwrapper import MonoDepthv2Wrapper
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["icl", "tum"],
    help="Dataset to use. Supported options:\n"
    " icl = Ground Truth odometry\n"
    " tum = Iterative Closest Point\n",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help="Path to the dataset directory",
)
parser.add_argument(
    "--odometry",
    type=str,
    default="gradicp",
    choices=["gt", "icp", "gradicp"],
    help="Odometry method to use. Supported options:\n"
    " gt = Ground Truth odometry\n"
    " icp = Iterative Closest Point\n"
    " gradicp (*default) = Differentiable Iterative Closest Point\n",
)
args = parser.parse_args()

def change_res(input, new_res, interpol = "bicubic"):
    bsize, ssize, height, width, csize = input.size(0), input.size(1), input.size(2), input.size(3), input.size(4)
    input = input.view([bsize*ssize, height, width, csize]).permute(0, 3, 1, 2)
    input = torch.nn.functional.interpolate(input = input, size = new_res, mode = interpol)
    input = input.permute(0, 2, 3, 1).view([bsize, ssize, new_res[0], new_res[1], csize])
    return input

def pose_loss(gt, pred, device):
    gt = gt.to(device)
    pred = pred.to(device)
    T_gt = gt[:, :, :3, -1:].view([-1, 3])
    T_pred = pred[:, :, :3, -1:].view([-1, 3])
    R_gt = gt[:, :, :3, :3].view([-1, 3, 3])
    R_pred = pred[:, :, :3, :3].view([-1, 3, 3])

    T_error = torch.square(T_gt - T_pred).sum([-1])
    R_ = torch.eye(3).view([1, 3, 3]).to(device) - torch.matmul(torch.transpose(R_gt, 1, 2), R_pred)
    R_error = torch.tensor([torch.trace(i) for i in R_], device=device)
    loss = torch.sqrt(T_error + 2 * torch.abs(R_error) /3)
    return loss

if __name__ == "__main__":
    # select device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    DepthPredictor = MonoDepthv2Wrapper(args, device)
    
    optim = Adam(DepthPredictor.parameters(), lr = 1e-4)

    # load dataset
    if args.dataset == "icl":
        dataset = ICL(args.dataset_path, seqlen=10, height=192, width=640)
    elif args.dataset == "tum":
        dataset = TUM(args.dataset_path, seqlen=10, height=192, width=640, sequences = "/media/hamza/DATA/Data/list.txt")
        # dataset = TUM(args.dataset_path, seqlen=10, height=192, width=640, sequences = ("rgbd_dataset_freiburg1_floor", "rgbd_dataset_freiburg1_desk2"))

    loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

    writer = SummaryWriter()

    epochs = 50
    losses = []
    counter = {"every": 0, "batch": 0, "detailed": 0}
    for e_idx in range(epochs):
        for batch_idx, (colors, depths, intrinsics, poses, *_) in enumerate(loader):
            colors = colors.to(device)
            depths = depths.to(device)
            intrinsics = intrinsics.to(device)
            poses = poses.to(device)

            # Hard coded
            batch_loss = 0
            pred_depths = []
            for pred_index in range(10):
                DepthPredictor.zero_grad()
                input_dict = {"device": device}
                input_dict["rgb"] = (colors[:, pred_index, ::] / 255.0).permute(0, 3, 1, 2)
                input_dict["rgb_ref"] = (colors[:, pred_index - 1, ::] / 255.0).permute(0, 3, 1, 2)
                input_dict["pose"] = torch.matmul(torch.inverse(poses[:, pred_index - 1, ::]), poses[:, pred_index, ::])
                input_dict["depth"] = depths[:, pred_index, ::].permute(0, 3, 1, 2)
                input_dict["intrinsic"] = intrinsics

                if pred_index == 0:
                    depth_predictions, loss = DepthPredictor.pred_loss_depth(input_dict)
                elif pred_index > 0:
                    depth_predictions, loss = DepthPredictor.pred_loss_reproj(input_dict)
                    batch_loss += loss.item() * 0.1
                    loss.backward()
                    optim.step()
                    writer.add_scalar("Depth/Perstep_loss", loss.item(), counter["every"])
                    counter["every"] +=1

                pred_depths.append(depth_predictions.permute(0, 2, 3, 1).unsqueeze(1))


            writer.add_scalar("Depth/Batchwise_loss", batch_loss, counter["batch"] )
            losses.append(batch_loss)
            counter["batch"] +=1

            pred_depths = torch.cat(pred_depths, dim= 1)
            # projection = get_indexed_projection_TUM(proj_from_index = 1, proj_to_index=0, rgbs = colors, depths = depths, intrinsic = intrinsics, poses = poses, device = device)

            colors = change_res(colors, (120, 160), interpol="bicubic")
            pred_depths = change_res(pred_depths,(120, 160), interpol="nearest")
            intrinsics[:, :, 0, :] = intrinsics[:, :, 0, :] * 160 / 640
            intrinsics[:, :, 1, :] = intrinsics[:, :, 1, :] * 120 / 192

            # SLAM

            if batch_idx ==0 or batch_idx % 25 == 0:
                print("Getting pose loss")
                rgbdimages = RGBDImages(colors, pred_depths, intrinsics, poses, channels_first=False, device=torch.device("cpu"))
                slam = PointFusion(odom=args.odometry, dsratio=4, device=torch.device("cpu"))
                pointclouds, recovered_poses = slam(rgbdimages)
                loss_p = pose_loss(poses, recovered_poses, device)
                writer.add_scalar("Pose/Batchwise_loss", loss_p.mean().item(), counter["detailed"] )
                counter["detailed"] += 1


            print("loss: {}".format(sum(losses[-5:])/ 5))


    # visualization
    # o3d.visualization.draw_geometries([pointclouds.open3d(0)])
    # o3d.visualization.draw_geometries([pointclouds.open3d(1)])