from argparse import ArgumentParser, RawTextHelpFormatter

import open3d as o3d
import torch
from torch.utils.data import DataLoader

from gradslam.datasets.icl import ICL
from gradslam.datasets.tum import TUM
from end2endslam.dataloader.kitti import KITTI
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.rgbdimages import RGBDImages
from end2endslam.loss_hamza.reprojection_loss import get_indexed_projection_TUM, image2image
from perception.monodepth2.monodepthwrapper import MonoDepthv2Wrapper
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image as pil_image
import numpy as np
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import imageio
import os

"""r Example run commands:

KITTI 
(small test dataset provided at https://drive.google.com/drive/folders/1i1ydf1YOVLCZmOkByr8vfwEbOP9hJDYR?usp=sharing )
pointfusion_brucker.py --dataset kitti --dataset_path "/home/matthias/data/kitti/data_odometry/dataset"
--odometry gt --sequences "/home/matthias/data/kitti/data_odometry/dataset/sequences.txt" --visualize True

TUM (leave TUM folder structure unchanged)
pointfusion_brucker.py --dataset kitti --dataset_path "/home/matthias/data/kitti/data_odometry/dataset"
--odometry gt --visualize True

"""

parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["icl", "tum", "kitti"],
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
parser.add_argument(
    "--sequences",
    type=str,
    default=None,
    help="Path to .txt file containing sequences. \n"
    " If it doesn't work for TUM, leave option out, it then takes all sequences in dataset_path"
)
parser.add_argument(
    "--visualize",
    type=bool,
    default=False,
    help="Depth and SLAM Map Visualization"
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

    DepthPredictor = MonoDepthv2Wrapper(device)
    
    optim = Adam(DepthPredictor.parameters(), lr = 1e-6)

    # load dataset
    if args.dataset == "icl":
        dataset = ICL(args.dataset_path, seqlen=10, height=192, width=640)
    elif args.dataset == "tum":
        dataset = TUM(args.dataset_path, seqlen=10, height=192, width=640, sequences=args.sequences)
    elif args.dataset == "kitti":
        dataset = KITTI(args.dataset_path, seqlen=10, height=192, width=640, sequences=args.sequences)

    loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

    writer = SummaryWriter()

    epochs = 50
    losses = []
    counter = {"every": 0, "batch": 0, "detailed": 0}
    for e_idx in range(epochs):
        for batch_idx, (colors, depths, intrinsics, poses, *_) in enumerate(loader):
            colors = colors.to(device)
            # Todo: clean since kitti has no gt depth
            if args.dataset == "tum":
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
                # Todo: clean
                if args.dataset == "tum":
                    input_dict["depth"] = depths[:, pred_index, ::].permute(0, 3, 1, 2)
                else:
                    input_dict["depth"] = None
                input_dict["intrinsic"] = intrinsics

                if pred_index == 0:
                    #Todo: clean, KITTI does not provide GT depth
                    if args.dataset == "kitti":
                        depth_predictions, loss = DepthPredictor.pred_loss_reproj(input_dict)
                    else:
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

            if batch_idx ==0 or batch_idx % 10 == 0:
                print("Getting pose loss")

                # With predicted depth
                rgbdimages = RGBDImages(colors, pred_depths, intrinsics, poses, channels_first=False, device=torch.device("cpu"))
                slam = PointFusion(odom=args.odometry, dsratio=4, device=torch.device("cpu"))
                pointclouds, recovered_poses = slam(rgbdimages)
                loss_p = pose_loss(poses, recovered_poses, device)
                writer.add_scalar("Pose/Batchwise_loss_pred", loss_p.mean().item(), counter["detailed"] )

                # Visualize (images stored in vis directory)
                if args.visualize:
                    if not os.path.exists("vis"):
                        os.makedirs("vis")
                    # SLAM Visualization
                    o3d.visualization.draw_geometries([pointclouds.open3d(0)])
                    # Color Visualization
                    vis_color = colors[0, 0, :, :, :].detach().cpu().numpy()
                    imageio.imwrite("vis/test_{}_{}_color.png".format(args.dataset, batch_idx), vis_color)
                    # Depth Visualization
                    vis_pred_depth = pred_depths[0, 0, :, :, :].detach().cpu().numpy()
                    vmax = np.percentile(vis_pred_depth, 95)
                    vmin = vis_pred_depth.min()
                    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                    pred_depth_np = (mapper.to_rgba(vis_pred_depth[:, :, 0])[:, :, :3] * 255).astype(np.uint8)
                    imageio.imwrite("vis/test_{}_{}_depth.png".format(args.dataset, batch_idx), pred_depth_np)

                # With ground truth depth
                if args.dataset == "tum":
                    depths = change_res(depths,(120, 160), interpol="nearest")
                    rgbdimages = RGBDImages(colors, depths, intrinsics, poses, channels_first=False, device=torch.device("cpu"))
                    slam = PointFusion(odom=args.odometry, dsratio=4, device=torch.device("cpu"))
                    pointclouds, recovered_poses = slam(rgbdimages)
                    loss_p = pose_loss(poses, recovered_poses, device)
                    writer.add_scalar("Pose/Batchwise_loss_gt", loss_p.mean().item(), counter["detailed"] )

                counter["detailed"] += 1


            print("loss: {}".format(sum(losses[-5:])/ 5))
