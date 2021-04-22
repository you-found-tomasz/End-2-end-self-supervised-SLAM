from argparse import ArgumentParser, RawTextHelpFormatter

import torch
from torch.utils.data import DataLoader

from gradslam.datasets.tum import TUM
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.pointclouds import Pointclouds
#from perception.SC_SfMLearner_Release.scsfmwrapper import SCSfmWrapper
from scsfmwrapper import SCSfmWrapper
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

"""r Example run commands:

(leave TUM folder structure unchanged):

Debug mode with visulizations saved to given path:
pointfusion_scsfm_brucker.py --dataset tum --dataset_path "/home/matthias/data/tum_test_1" --odometry icp 
--loss "depth_consistency"
--debug_path "/home/matthias/data/"
Plain mode:
pointfusion_scsfm_brucker.py --dataset tum --dataset_path "/home/matthias/data/tum_test_1" --odometry icp 
--loss "depth_consistency"


"""

parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["tum"],
    help="Dataset to use. Supported options:\n"
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
    "--debug_path",
    type=str,
    default=None,
    help="Debug, Depth and SLAM Map Visualization"
)
parser.add_argument(
    "--seq_length",
    type=int,
    default=10,
    help="SLAM Sequence Length"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help="SLAM Sequence Length"
)
parser.add_argument(
    "--loss",
    type=str,
    default="depth_consistency",
    choices=["gt_depth", "depth_consistency", "reprojection"]
)
args = parser.parse_args()

if __name__ == "__main__":
    # select device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Initialize DepthPrediction Network
    DepthPredictor = SCSfmWrapper(device)
    optim = Adam(DepthPredictor.parameters(), lr = 1e-6)

    # load dataset
    if args.dataset == "tum":
        # doesn't work with 120 / 160, probably has something to do with upsampling / downsampling in ResNet
        dataset = TUM(args.dataset_path, seqlen=args.seq_length, height=480, width=640, sequences=None)

    # get data
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    writer = SummaryWriter()

    # Training
    epochs = 50
    losses = []
    counter = {"every": 0, "batch": 0, "detailed": 0}
    for e_idx in range(epochs):
        # TODO: remove gt depth dependency
        for batch_idx, (colors, depths, intrinsics, poses, *_) in enumerate(loader):
            colors = colors.to(device)
            depths = depths.to(device)
            intrinsics = intrinsics.to(device)
            poses = poses.to(device)

            # Hard coded
            batch_loss = 0
            pred_depths = []

            # Initialize SLAM and pointclouds
            slam = PointFusion(odom=args.odometry, dsratio=4, device=device)
            pointclouds = Pointclouds(device=device)
            prev_frame = None

            # Scale intrinsics since SLAM works on downsampled images
            intrinsics[:, :, 0, :] = intrinsics[:, :, 0, :] * 160 / 640
            intrinsics[:, :, 1, :] = intrinsics[:, :, 1, :] * 120 / 480

            # Iterate over frames in Sequence
            for pred_index in range(args.seq_length):
                DepthPredictor.zero_grad()

                # get input tensors
                input_dict = {"device": device}
                input_dict["rgb"] = (colors[:, pred_index, ::] / 255.0).permute(0, 3, 1, 2)
                input_dict["rgb_ref"] = (colors[:, pred_index - 1, ::] / 255.0).permute(0, 3, 1, 2)
                # Initial poses are necessary. We use identity here, could also use gt
                input_dict["pose"] = torch.eye(4, device=device).view(1, 4, 4).repeat(args.batch_size, 1, 1)
                # input_dict["pose"] = torch.matmul(torch.inverse(poses[:, pred_index - 1, ::]), poses[:, pred_index, ::])
                input_dict["depth"] = depths[:, pred_index, ::].permute(0, 3, 1, 2)
                input_dict["intrinsic"] = intrinsics

                # predict, backprop, and optimize (depth consistency loss is computed every frame of sequence)
                if args.loss == "depth_consistency":
                    depth_predictions, loss, slam, pointclouds, prev_frame = DepthPredictor.pred_loss_depth_consistency(
                        input_dict, slam, pointclouds, prev_frame, args.debug_path)
                elif args.loss == "reprojection":
                    depth_predictions, loss = DepthPredictor.pred_loss_reproj(input_dict)
                elif args.loss == "gt_depth":
                    depth_predictions, loss = DepthPredictor.pred_loss_depth(input_dict)

                batch_loss += loss.item() * 0.1
                loss.backward()
                optim.step()
                print("Batch_idx: {} / Seq: {} / Loss ({}): {}".format(batch_idx, pred_index, args.loss, loss))

                # Tensorboard
                writer.add_scalar("Depth/Perstep_loss", loss.item(), counter["every"])
                counter["every"] +=1

                pred_depths.append(depth_predictions.permute(0, 2, 3, 1).unsqueeze(1))

            writer.add_scalar("Depth/Batchwise_loss", batch_loss, counter["batch"] )
            losses.append(batch_loss)
            counter["batch"] +=1

            pred_depths = torch.cat(pred_depths, dim= 1)


