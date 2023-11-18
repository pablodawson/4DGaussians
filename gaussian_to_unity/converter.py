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
                               pos_format: str = "Norm11", 
                               basepath: str ="/", idx=1) -> None:

    timestart = tm.time()
    means3D_sorted = means3d[order_indexes].cpu().numpy().copy()

    rotations_to_save, scales_linealized= linealize(rotations[order_indexes].cpu().numpy().copy(), 
                                                  scales[order_indexes].cpu().numpy().copy())
    if debug:
        print("linealization time:", tm.time()-timestart)

    timestart = tm.time()
    chunkSize = 256
    means3D_to_save, scales_to_save, means_chunks, scale_chunks = create_chunks(means3D_sorted, scales_linealized, means3d.shape[0], chunkSize)
    
    if debug:
        print("chunk creation time:", tm.time()-timestart)
    
    timestart = tm.time()
    create_positions_asset(means3D_to_save, basepath, format=pos_format, idx= idx)

    if debug:
        print("create_positions_asset time:", tm.time()-timestart)
    
    #timestart = tm.time()
    #create_others_asset(rotations_to_save, scales_to_save, sh_index, basepath, scale_format="Norm11", idx= idx)
   # print("create_others_asset time:", tm.time()-timestart)

    timestart = tm.time()
    create_chunks_asset(means_chunks, scale_chunks, basepath, idx= idx)

    if debug:
        print("create_chunks_asset time:", tm.time()-timestart)

def static_data_to_unity(scales: torch.tensor, rotations: torch.tensor, order_indexes: np.array):
    