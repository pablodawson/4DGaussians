ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1,2,4,8],
    defor_depth = 2,
    net_width = 256,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.001,
    render_process=False,

)
OptimizationParams = dict(
    dataloader=True,
    iterations = 35_000,
    coarse_iterations = 3000,
    densify_until_iter = 15_000,
    opacity_reset_interval = 6000,

    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,

    # LightGaussian pruning
    prune_iterations = [500, 24_000],
    prune_percent = 0.5,
    v_pow = 0.1,
    prune_decay = 0.7
)
