import time as tm
import numpy as np
from gaussian_to_unity.utils import *
import torch

# Reorder the points using Morton order. Returns the indexes of the points in the new order.
def get_order(means3d: torch.tensor) -> np.array:

    # Calculate bounds
    bounds_min, bounds_max = calculate_bounds(means3d.cpu().numpy())
    
    # Morton sort
    order = reorder_morton_job(bounds_min, 1.0/(bounds_max - bounds_min), means3d)
    order_indexes = order[:,1].tolist()
    
    return order_indexes

# Convert the splats of a certain timestep to the Unity format. Each run appends a new frame to the asset.
def gaussian_timestep_to_unity(means3d: torch.tensor, 
                               scales: torch.tensor, 
                               rotations: torch.tensor, 
                               order_indexes: np.array, 
                               debug: bool =False, 
                               args = None, 
                               basepath: str ="/", idx=1) -> None:

    #test_xyz = pc.get_xyz[order_indexes].cpu().numpy()
    timestart = tm.time()
    means3D_sorted = means3d[order_indexes].cpu().numpy().copy()

    
    if debug:
        print("linealization time:", tm.time()-timestart)

    timestart = tm.time()
    chunkSize = 256

    means3D_to_save, means_chunks = create_chunks(means3D_sorted, means3d.shape[0], chunkSize)
    
    
    if debug:
        print("chunk creation time:", tm.time()-timestart)
    
    timestart = tm.time()
    create_positions_asset(means3D_to_save, basepath, format=args.pos_format, idx= idx)

    if debug:
        print("create_positions_asset time:", tm.time()-timestart)

    timestart = tm.time()

    rotations_to_save, scales_linealized= linealize(rotations[order_indexes].cpu().numpy().copy(), 
                                                scales[order_indexes].cpu().numpy().copy())
    
    global scale_chunks

    if ((args.include_others) or idx == 0):
        scales_to_save, scale_chunks = create_chunks(scales_linealized, means3d.shape[0], chunkSize)

    timestart = tm.time()
    create_chunks_asset(means_chunks, scale_chunks, basepath, idx= idx)

    if debug:
        print("create_chunks_asset time:", tm.time()-timestart)

    if (args.include_others):
        create_others_asset(rotations_to_save, scales_to_save, basepath, scale_format=args.scale_format, idx= idx)

    if debug:
        print("create_others_asset time:", tm.time()-timestart)

def gaussian_static_data_to_unity(splat_count: int,
                        scales: torch.tensor, 
                        rotations: torch.tensor, 
                        dc: torch.tensor,
                        shs: torch.tensor,
                        opacity: torch.tensor,
                        order_indexes: np.array,
                        args = None, 
                        basepath: str ="/"):
    
    print("Saving static data")

    if (not args.include_others):

        chunkSize = 256
        timestart = tm.time()
        rotations_to_save, scales_linealized= linealize(rotations[order_indexes].cpu().numpy().copy(), 
                                                    scales[order_indexes].cpu().numpy().copy())
        scales_to_save, _ = create_chunks(scales_linealized, splat_count, chunkSize)

        create_others_asset(rotations_to_save, scales_to_save, basepath, scale_format=args.scale_format)

        print("create_others_asset time:", tm.time()-timestart)
        
    chunkSize = 256
    
    # Morton reorder
    dc = dc.cpu().numpy()[order_indexes]
    shs = shs.cpu().numpy()[order_indexes]
    opacity = opacity.cpu().numpy()[order_indexes]

    timestart = tm.time()
    # Cluster shs
    shs, _ = cluster_shs(shs, sh_format=args.sh_format)

    # Linealize colors
    dc, opacity = linealize_colors(dc, opacity)

    print("cluster shs and linealize colors time:", tm.time()-timestart)

    timestart = tm.time()
    # dc: N, 1, 3 to N, 3
    dc = dc.squeeze(1) 
    
    shs = shs[:, 1:, :]
    
    col = np.concatenate((dc, opacity), axis=1)
    col_to_save, col_chunks = create_chunks(col, splat_count, chunkSize)
    shs_to_save, shs_chunks = create_chunks(shs, splat_count, chunkSize, True)

    shs_chunks = shs_chunks.squeeze(2)
    print("create_chunks time:", tm.time()-timestart)
    
    timestart = tm.time()
    create_colors_asset(splat_count, col_to_save, color_format=args.col_format, basepath=basepath)
    print("create_colors_asset time:", tm.time()-timestart)

    timestart = tm.time()
    create_sh_asset(splat_count, shs_to_save, sh_data_format=args.sh_format ,basepath=basepath)
    print("create_sh_asset time:", tm.time()-timestart)

    timestart = tm.time()
    create_chunks_static_asset(col_chunks, shs_chunks, basepath)

    print("create_chunks_static_asset time:", tm.time()-timestart)