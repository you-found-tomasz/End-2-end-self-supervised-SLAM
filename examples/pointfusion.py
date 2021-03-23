from argparse import ArgumentParser, RawTextHelpFormatter

import open3d as o3d
import torch
from torch.utils.data import DataLoader

from gradslam.datasets.icl import ICL
from gradslam.datasets.tum import TUM
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.rgbdimages import RGBDImages
from loss_hamza.reprojection_loss import get_indexed_projection_TUM
from depth.monodepth2.monodepthwrapper import MonoDepthv2Wrapper
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam

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
    default="gt",
    choices=["gt", "icp", "gradicp"],
    help="Odometry method to use. Supported options:\n"
    " gt = Ground Truth odometry\n"
    " icp = Iterative Closest Point\n"
    " gradicp (*default) = Differentiable Iterative Closest Point\n",
)
args = parser.parse_args()

if __name__ == "__main__":
    # select device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load dataset
    if args.dataset == "icl":
        dataset = ICL(args.dataset_path, seqlen=10, height=192, width=640)
    elif args.dataset == "tum":
        dataset = TUM(args.dataset_path, seqlen=10, height=192, width=640)

    DepthPredictor = MonoDepthv2Wrapper(args, device)

    loader = DataLoader(dataset=dataset, batch_size=4)
    # Todo: optim with  the mono depth parameters
    
    optim = Adam(DepthPredictor.parameters(), lr = 1e-4)


    epochs = 25
    losses = []
    for e_idx in range(epochs):
        for batch_idx, (colors, depths, intrinsics, poses, *_) in enumerate(loader):

            colors = colors.to(device)
            depths = depths.to(device)
            intrinsics = intrinsics.to(device)
            poses = poses.to(device)

            # Hard coded
            batch_loss = 0
            for pred_index in range(10):
                DepthPredictor.zero_grad()
                rgbs_curr = (colors[:, pred_index, ::] / 255.0).permute(0, 3, 1, 2)
                depths_curr = depths[:, pred_index, ].permute(0, 3, 1, 2)
                depth_predictions, loss = DepthPredictor.get_loss_depth(rgbs_curr, depths_curr)
                batch_loss += loss["abs"].item() * 0.1
                loss["abs"].backward()
                optim.step()
            
            losses.append(batch_loss)
            print("loss: {}".format(sum(losses[-5:])/ 5))



    # rgbdimages = RGBDImages(colors, depths, intrinsics, poses, channels_first=False)
    # projection = get_indexed_projection_TUM(proj_from_index = 1, proj_to_index=0, rgbs = colors, depths = depths, intrinsic = intrinsics, poses = poses, device = device)

    #     # SLAM
    # slam = PointFusion(odom=args.odometry, dsratio=4, device=device)
    # pointclouds, recovered_poses = slam(rgbdimages)

    # visualization
    # o3d.visualization.draw_geometries([pointclouds.open3d(0)])
    # o3d.visualization.draw_geometries([pointclouds.open3d(1)])