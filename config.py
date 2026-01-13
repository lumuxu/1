class FLAGES(object):
    """Training configuration.

    For upsampled MS (same spatial size as PAN):
        pan_size == ms_size
        ratio = 1

    For classic PAN/MS ratio (e.g. PAN is 4x MS):
        pan_size = ratio * ms_size
        ratio = 4

    NOTE: If you change the source TIFFs, please delete the existing H5 file
    (data_path) or change data_path to a new filename, otherwise the cached
    patches will be reused.
    """

    # Patch size (H=W)
    pan_size = 32
    ms_size = 32

    # MS band count
    num_spectrum = 8

    # Spatial ratio (PAN / MS). If MS is already upsampled to PAN size, set 1.
    ratio = 1

    # Patch stride
    stride = 16

    norm = True

    batch_size = 32
    lr = 0.0001
    decay_rate = 0.99
    decay_step = 10000

    # Optional training knobs for PanGan (train.py will pick these up if set).
    d_lr = None
    lambda_hp = 5.0
    lambda_spec = 1.0
    lambda_adv_spatial = 1.0
    lambda_adv_spectrum = 1.0
    adv_warmup_iters = 2000
    adv_ramp_iters = 8000
    adv_weight_max = 1.0
    residual_scale = 0.1
    beta1 = 0.5
    beta2 = 0.999
    grad_clip_norm = 5.0

    # Root folder containing MS/ and Pan/ subfolders with TIFFs.
    img_path = './data/source_data'

    # Cached patch dataset
    data_path = './data/train/train_tif.h5'

    log_dir = './log'
    model_save_dir = './model'

    is_pretrained = False

    iters = 5000
    model_save_iters = 1000
    valid_iters = 10
    
    loss_log_iters = 10            # 每 10 step 记录一次 loss（默认 1）
    loss_plot_max_points = 20000   # 画图时最多取多少点（默认 20000，自动下采样）
    best_k = 5                     # 输出 TopK（默认 5）
    loss_log_flush_every = 200     # CSV 每写多少行 flush 一次（默认 200）
