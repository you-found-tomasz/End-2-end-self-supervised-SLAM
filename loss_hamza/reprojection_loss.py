import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class BackprojectDepth(nn.Module):
    def __init__(self, cfg, batch_size):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = cfg.height
        self.width = cfg.width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)
        self.ones2 = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width).to(cfg.device),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(self.batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1).to(cfg.device),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones2], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, cfg, batch_size):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = cfg.height
        self.width = cfg.width
        self.eps = 1e-7

    def forward(self, points, K, T):
        P = torch.matmul(K, T[:,:3,:])[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords

class Cam2Cam(nn.Module):
    """Layer that transforms one camera to another camera
    """
    def __init__(self, cfg):
        super(Cam2Cam, self).__init__()
        self.cam2PC = BackprojectDepth(cfg)
        self.PC2cam = Project3D(cfg)

    def forward(self, img, depth, T, K = None, inv_K = None):
        cam_points = self.cam2PC(depth, inv_K)
        pix_coords = self.PC2cam(cam_points, K, T)
        out_img = F.grid_sample(img, pix_coords, padding_mode='border')
        return out_img

    @staticmethod
    def transform(img, depth, K1 = None, K2= None, E1= None, E2 = None, device=None):
        batch_size = img.size()[0]
        width = img.size()[-1]
        height = img.size()[-2]
        eps = 1e-7

        meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        id_coords = nn.Parameter(torch.from_numpy(id_coords),requires_grad=False)
        ones = nn.Parameter(torch.ones(batch_size, 1, height * width),requires_grad=False)
        ones2 = nn.Parameter(torch.ones(batch_size, 1, height * width).to(device), requires_grad=False)

        pix_coords = torch.unsqueeze(torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0)
        pix_coords = pix_coords.repeat(batch_size, 1, 1)
        pix_coords = nn.Parameter(torch.cat([pix_coords, ones], 1).to(device), requires_grad=False)

        cam_points = torch.matmul(torch.inverse(K1)[:, :3, :3], pix_coords)
        cam_points = depth.view(batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, ones2], 1)


        T = torch.matmul(E2, torch.inverse(E1))

        P = torch.matmul(K2, T)

        cam_points = torch.matmul(P, cam_points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + eps)
        pix_coords = pix_coords.view(batch_size, 2, height, width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= width - 1
        pix_coords[..., 1] /= height - 1
        pix_coords = (pix_coords - 0.5) * 2
        out_img = F.grid_sample(img, pix_coords, padding_mode='border')
        return out_img

def get_indexed_projection_TUM(proj_from_index, proj_to_index, rgbs, depths, intrinsic, poses, device):
    """Get the reprojection from projection index to the reference index [0] taken as a default here, we can easily change this to some other as well

    Args:
        projection_index ([int]): [Index to reproject from]
        rgbs ([Tensor]): [Images tensor from the TUM output scheme [b, s, h, w, c]]
        depths ([Tensor]): [Depths for the images [b, s, h, w, 1]]
        intrinsic ([Tensor]): [Intrinsic matrix [b, 0, 4, 4]]
        poses ([Tensor]): [Poses referenced from Image at index 0 [b, s, 4, 4]]
        device ([torch.device]): [CPU or GPU]
    """
    # Change this for plotting etc
    debug = False

    rgb_i = rgbs[:, proj_from_index, ::].permute(0, 3, 1, 2)
    depths_i = depths[:, proj_from_index, ::]
    intrinsic_i = intrinsic[:, 0, ::]
    poses_i = torch.matmul(torch.inverse(poses[:, proj_to_index, ::]), poses[:, proj_from_index, ::])
    images_reprojected = Cam2Cam.transform(rgb_i, depths_i, intrinsic_i, intrinsic_i, poses_i, poses_i, device=device)

    if debug:
        aux = np.clip(rgb_i[0, ::].permute(1, 2, 0).cpu().detach().numpy(), 0, 255).astype("uint8")
        ref = np.clip(rgbs[0, proj_to_index, ::].cpu().detach().numpy(), 0, 255).astype("uint8")
        reproj = np.clip(images_reprojected[0, ::].permute(1, 2, 0).cpu().detach().numpy(), 0, 255).astype("uint8")

        plt.imshow(np.hstack([ref, aux, reproj]))
        plt.show()

    return images_reprojected
