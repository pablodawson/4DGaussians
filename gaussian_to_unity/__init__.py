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
import time as tm


def get_order(pc : GaussianModel):

    means3D = pc.get_xyz

    # Calculate bounds
    bounds_min, bounds_max = calculate_bounds(pc.get_xyz.cpu().numpy())
    
    # Morton sort
    order = reorder_morton_job(bounds_min, 1.0/(bounds_max - bounds_min), means3D)
    order_indexes = order[:,1].tolist()
    
    return order_indexes


def save_frame(viewpoint_camera, pc : GaussianModel, pipe, scaling_modifier = 1.0, 
               stage="fine", order_indexes=None, basepath = "output", 
               idx=0, pos_format="Norm11"):
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

    timestart  = tm.time()
    if stage == "coarse" :
        means3D_deform, scales_deform, rotations_deform, opacity_deform = means3D, scales, rotations, opacity
    else:
        means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
                                                                         rotations[deformation_point], opacity[deformation_point],
                                                                         time[deformation_point])
    print("deformation MLP time:", tm.time()-timestart)

    # print(time.max())
    with torch.no_grad():
        pc._deformation_accum[deformation_point] += torch.abs(means3D_deform-means3D[deformation_point])

    timestart = tm.time()
    means3D_final = torch.zeros_like(means3D)
    rotations_final = torch.zeros_like(rotations)
    scales_final = torch.zeros_like(scales)
    means3D_final[deformation_point] =  means3D_deform
    rotations_final[deformation_point] =  rotations_deform
    scales_final[deformation_point] =  scales_deform
    means3D_final[~deformation_point] = means3D[~deformation_point]
    rotations_final[~deformation_point] = rotations[~deformation_point]
    scales_final[~deformation_point] = scales[~deformation_point]

    print("array indexing time:", tm.time()-timestart)
    # Keep the sorted order of the points

    timestart = tm.time()
    means3D_sorted = means3D_final[order_indexes].cpu().numpy().copy()

    rotations_to_save, scales_linealized= linealize(rotations_final[order_indexes].cpu().numpy().copy(), 
                                                  scales_final[order_indexes].cpu().numpy().copy())
    
    print("linealization time:", tm.time()-timestart)

    timestart = tm.time()
    chunkSize = 256
    means3D_to_save, scales_to_save, means_chunks, scale_chunks = create_chunks(means3D_sorted, scales_linealized, means3D.shape[0], chunkSize)
    sh_index = None
    
    # Debug reconstruction
    debug = True

    if (debug):
        pos_debug = means3D_to_save[0:chunkSize].copy()
        min_pos_chunk = means_chunks[0][0]
        max_pos_chunk = means_chunks[0][1]

        pos_recon = min_pos_chunk + pos_debug * (max_pos_chunk - min_pos_chunk)

        # Ensure correct reconstruction
        print((pos_recon[0:20] - means3D_sorted[0:20] <= 1e-2).all())

    print("chunk creation time:", tm.time()-timestart)

    timestart = tm.time()
    create_positions_asset(means3D_to_save, basepath, format=pos_format, idx= idx)
    print("create_positions_asset time:", tm.time()-timestart)
    
    #timestart = tm.time()
    #create_others_asset(rotations_to_save, scales_to_save, sh_index, basepath, scale_format="Norm11", idx= idx)
   # print("create_others_asset time:", tm.time()-timestart)

    timestart = tm.time()
    create_chunks_asset(means_chunks, scale_chunks, basepath, idx= idx)
    print("create_chunks_asset time:", tm.time()-timestart)
    