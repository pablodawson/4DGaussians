#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from gaussian_to_unity.utils import *



def create_initial_asset(pc : GaussianModel, pipe, scaling_modifier = 1.0):

    means3D = pc.get_xyz

    # Calculate bounds
    bounds_min, bounds_max = calculate_bounds(pc.get_xyz.cpu().numpy())
    
    # Morton sort
    order = reorder_morton_job(bounds_min, 1.0/(bounds_max - bounds_min), means3D)
    order_indexes = order[:,1].tolist()
    
    means3D_sorted = means3D[order_indexes].cpu().numpy()
    path_pos = "pos.bytes"
    
    chunkSize = 256
    means3D_sorted = create_chunks(means3D_sorted, means3D.shape[0], chunkSize)

    create_positions_asset(means3D_sorted, path_pos, format="Norm11")

    return means3D_sorted, order[:,1].tolist()


def save_frame(viewpoint_camera, pc : GaussianModel, pipe, scaling_modifier = 1.0, stage="fine", order_indexes=None, save_name="pos.bytes"):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    means3D = pc.get_xyz
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    opacity = pc._opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    if stage == "coarse" :
        means3D_deform, scales_deform, rotations_deform, opacity_deform = means3D, scales, rotations, opacity
    else:
        means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
                                                                         rotations[deformation_point], opacity[deformation_point],
                                                                         time[deformation_point])
    # print(time.max())
    with torch.no_grad():
        pc._deformation_accum[deformation_point] += torch.abs(means3D_deform-means3D[deformation_point])

    means3D_final = torch.zeros_like(means3D)
    means3D_final[deformation_point] =  means3D_deform
    means3D_final[~deformation_point] = means3D[~deformation_point]

    # Keep the sorted order of the points
    means3D_to_save = means3D_final[order_indexes].cpu().numpy()
    
    chunkSize = 256
    means3D_to_save = create_chunks(means3D_to_save, means3D.shape[0], chunkSize)
    create_positions_asset(means3D_to_save, save_name, format="Norm11")