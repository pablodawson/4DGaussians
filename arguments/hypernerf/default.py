ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1,2,4],
    defor_depth = 1,
    net_width = 128,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    render_process=False
)
OptimizationParams = dict(
    # dataloader=True,
    iterations = 14_000,
    batch_size=2,
    coarse_iterations = 3_000,
    densify_until_iter = 10_000,
    opacity_reset_interval = 300_000,
    prune_iterations = [500, 10_000],
    prune_percent = 0.5,
    v_pow = 0.1,
    prune_decay = 0.7,
    depth_model = 'zoe',
    # Depth regularization
    regularize_depth = False,
    regularize_depth_start = 1,
    regularize_depth_end = 7_000,
    lambda_depth = 0.2
    # grid_lr_init = 0.0016,
    # grid_lr_final = 16,
    # opacity_threshold_coarse = 0.005,
    # opacity_threshold_fine_init = 0.005,
    # opacity_threshold_fine_after = 0.005,
    # pruning_interval = 2000
)