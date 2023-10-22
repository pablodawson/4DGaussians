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
from gaussian_to_unity import save_frame, get_order
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)


def render_set(model_path, name, iteration, views, gaussians, pipeline):
    
    save_path = os.path.join(model_path, "render_unity")
    makedirs(save_path, exist_ok=True)
    
    order = get_order(gaussians)

    idx2= 0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:
            time1 = time()
        if (idx % args.save_interval) == 0:
            idx2+=1
            save_frame(view, gaussians, pipeline, order_indexes=order, basepath=save_path, idx=idx2)

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
    parser.add_argument("--save_interval", default=10)
    parser.add_argument("--configs", type=str, default="arguments/hypernerf/default.py")
    args = get_combined_args(parser)
    
    args.model_path = "output/cookie"
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)