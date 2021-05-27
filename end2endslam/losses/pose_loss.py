import torch
from pytorch3d.transforms import matrix_to_euler_angles

def pose_loss_wrapper(args,input_dict):
    """ Wraps pose_diff_loss by accepting the input_dict and choosing the poses to compare
    """
    pose_1 = input_dict["slam_rel_poses"]
    pose_2 = input_dict["gt_rel_poses"]
    return pose_diff_loss(pose_1,pose_2)

def pose_loss_unified(pose_1,pose_2,args):
    """Computes different and combined pose losses"""
    loss = {"pose_com": 0}

    loss['pose_rot'],loss['pose_trans'] = pose_diff_loss(pose_1,pose_2)

    loss['pose_com'] = args.loss_pose_trans_factor*loss['pose_trans'] + args.loss_pose_rot_factor*loss['pose_rot']

    return loss


def pose_diff_loss(pose_1,pose_2):
    """ computes difference between two different poses
    
    Decomposes poses into translation and rotation angles, computes losses
    between each component

    Args:
        pose_1: 4x4 tensor, pose matrix homogeneous coords
        pose_2: 4x4 tensor,pose matrix homogeneous coords
    Returns:
        scalar losses, which represents difference between the two poses
        rot_loss: MSE difference between angle vectors
        trans_loss: MSE difference between translation vectors
    """

    mse_loss = torch.nn.MSELoss()
    #loss = mse_loss(pose_1,pose_2)

    # decompose poses into translation and rotation angles

    pose_1_r = pose_1[:,:,:3,:3]
    pose_1_angl = matrix_to_euler_angles(pose_1_r,"XYZ")
    pose_1_t = pose_1[:,:,:3,3]

    pose_2_r = pose_2[:,:,:3,:3]
    pose_2_angl = matrix_to_euler_angles(pose_2_r,"XYZ")
    pose_2_t = pose_2[:,:,:3,3]

    #compute MSE loss for angle vector and translation vector
    
    rot_loss = mse_loss(pose_1_angl,pose_2_angl)
    trans_loss = mse_loss(pose_1_t,pose_2_t)
    
    return  rot_loss, trans_loss