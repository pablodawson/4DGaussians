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
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_to_unity import save_frame
from gaussian_to_unity.converter import get_order
from gaussian_to_unity.utils import create_one_file, create_deleted_mask
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)


def render_set(model_path, name, iteration, views, gaussians, pipeline):
    
    save_path = os.path.join(model_path, "unity_format/")
    makedirs(save_path, exist_ok=True)
    
    order = get_order(gaussians.get_xyz)
    
    if (args.deleted_path is not None):
        mask = create_deleted_mask(args.deleted_path)
        gaussians.prune_points_render(mask, order)
        order = get_order(gaussians.get_xyz) # Reorder the points after pruning
    
    idx2= 0

    for idx, view in enumerate(tqdm(views, desc="Conversion progress")):
        if idx == 0:
            time1 = time()
        if (idx % args.save_interval) == 0:
            save_frame(view, gaussians, pipeline, order_indexes=order, basepath=save_path, idx=idx2, args=args)
            idx2+=1
    
    # Create json with metadata

    splat_count = gaussians.get_xyz.cpu().numpy().shape[0]
    chunk_count = (splat_count+args.chunk_size-1) // args.chunk_size
    
    #create_json(save_path, splat_count, chunk_count, args.pos_format, args.save_interval, args.fps, len(views))
    create_one_file(save_path, splat_count=splat_count, chunk_count=chunk_count, frame_time=args.save_interval/args.fps, args=args)

    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str, default="arguments/dynerf/default.py")
    
    parser.add_argument("--save_interval", default=2, type=int)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=30)
    
    parser.add_argument("--pos-format", type=str, default="Norm11")
    parser.add_argument("--scale_format", type=str, default="Norm11")
    parser.add_argument("--sh_format", type=str, default="Norm6")
    parser.add_argument("--col_format", type=str, default="Norm8x4")

    parser.add_argument("--save_name", type=str, default="scene")
    parser.add_argument("--include_others", action="store_true")
    
    parser.add_argument("--pos_offset", type=list, default=[0,0,0.35])
    parser.add_argument("--rot_offset", type=list, default=[0,180,0])
    parser.add_argument("--scale", type=list, default=[0.55,0.55,0.55])

    # Edits
    parser.add_argument("--deleted_path", type=str, default=None)
    
    args = get_combined_args(parser)
    
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)