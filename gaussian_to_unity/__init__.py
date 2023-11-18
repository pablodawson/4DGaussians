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
from gaussian_to_unity.converter import gaussian_timestep_to_unity, static_data_to_unity
import time as tm




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

    means3D_final = torch.zeros_like(means3D)
    rotations_final = torch.zeros_like(rotations)
    scales_final = torch.zeros_like(scales)
    means3D_final[deformation_point] =  means3D_deform
    rotations_final[deformation_point] =  rotations_deform
    scales_final[deformation_point] =  scales_deform
    means3D_final[~deformation_point] = means3D[~deformation_point]
    rotations_final[~deformation_point] = rotations[~deformation_point]
    scales_final[~deformation_point] = scales[~deformation_point]
    
    
    
    # Create Unity compatible frames for each gaussian state (only position at the moment)
    gaussian_timestep_to_unity(means3D_final, scales_final, rotations_final, order_indexes, debug=True, 
                               pos_format=pos_format, basepath=basepath, idx=idx)