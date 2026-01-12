import os
import time
import numpy as np
import tensorflow as tf
import cv2
import tifffile

from PanGan import PanGan

# If you want to force CPU (same as original script)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 1, 'Batch size')
tf.app.flags.DEFINE_integer('num_spectrum', 8, 'Number of MS bands')
tf.app.flags.DEFINE_integer('ratio', 1, 'PAN/MS ratio. If MS has been upsampled, set 1.')

tf.app.flags.DEFINE_string('model_path', './model/Generator-107000', 'Path to saved generator checkpoint')
tf.app.flags.DEFINE_string('test_path', './data/test_gt', 'Test data root path')
tf.app.flags.DEFINE_string('result_path', './result', 'Output folder')

# Folder names under test_path (keep same as original repo by default)
tf.app.flags.DEFINE_string('ms_dir', 'lrms', 'Subfolder for MS images (TIFF)')
tf.app.flags.DEFINE_string('pan_dir', 'pan', 'Subfolder for PAN images (TIFF)')
tf.app.flags.DEFINE_string(
    'pan_name_replace',
    '_mul,_pan',
    'Map MS filename to PAN filename via "from,to" replacement. Empty to keep same name.',
)
tf.app.flags.DEFINE_string(
    'input_layout',
    'auto',
    'Input layout for TIFF arrays: auto, hwc, or chw.',
)
tf.app.flags.DEFINE_string(
    'output_layout',
    'hwc',
    'Output layout before saving: auto, hwc, or chw.',
)

# Save dtype
# - If True: save output as uint16 when input MS is uint16; otherwise uint8
# - If False: always save uint8
# (Output is always scaled from [-1,1] back to integer range)
tf.app.flags.DEFINE_boolean('save_like_input_dtype', True, 'Save output dtype same as MS input (uint8/uint16)')


def _ensure_hwc(arr: np.ndarray) -> np.ndarray:
    """Ensure TIFF array is HWC."""
    layout = FLAGS.input_layout.lower()
    if layout == 'hwc':
        return arr
    if layout == 'chw':
        if arr.ndim == 2:
            return arr[:, :, None]
        if arr.ndim != 3:
            raise ValueError(f"Unsupported TIFF shape: {arr.shape}")
        return np.transpose(arr, (1, 2, 0))
    if layout != 'auto':
        raise ValueError('input_layout must be one of: auto, hwc, chw')

    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported TIFF shape: {arr.shape}")

    c_first, h, w = arr.shape
    if c_first <= 32 and c_first < h and c_first < w:
        return np.transpose(arr, (1, 2, 0))
    # Some TIFFs store as (H, C, W)
    h_first, c_mid, w_last = arr.shape
    if c_mid <= 32 and c_mid < h_first and c_mid < w_last:
        return np.transpose(arr, (0, 2, 1))
    return arr


def _robust_percentile_normalize_to_minus1_1(img: np.ndarray, p_low=0.2, p_high=99.8) -> np.ndarray:
    """Match training normalization: per-channel percentile scaling to [-1, 1]."""
    img_f = img.astype(np.float32)

    # special case: [0,1] floats
    if np.issubdtype(img.dtype, np.floating):
        vmin = float(np.nanmin(img_f))
        vmax = float(np.nanmax(img_f))
        if vmin >= 0.0 and vmax <= 1.0:
            return img_f * 2.0 - 1.0

    if img_f.ndim == 2:
        img_f = img_f[:, :, None]

    out = np.empty_like(img_f, dtype=np.float32)
    for c in range(img_f.shape[2]):
        ch = img_f[:, :, c]
        lo = np.nanpercentile(ch, p_low)
        hi = np.nanpercentile(ch, p_high)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
            lo = float(np.nanmin(ch))
            hi = float(np.nanmax(ch))
            if hi <= lo + 1e-6:
                out[:, :, c] = 0.0
                continue
        ch_n = (ch - lo) / (hi - lo)
        ch_n = np.clip(ch_n, 0.0, 1.0)
        out[:, :, c] = ch_n * 2.0 - 1.0

    return out


def _ensure_output_hwc(arr: np.ndarray) -> np.ndarray:
    layout = FLAGS.output_layout.lower()
    if layout == 'hwc':
        if arr.ndim == 3:
            c = int(getattr(FLAGS, 'num_spectrum', 0))
            if c > 0:
                if arr.shape[0] == c and arr.shape[2] != c:
                    return np.transpose(arr, (1, 2, 0))
                if arr.shape[1] == c and arr.shape[2] != c:
                    return np.transpose(arr, (0, 2, 1))
        return arr
    if layout == 'chw':
        if arr.ndim != 3:
            raise ValueError(f"Unsupported output shape: {arr.shape}")
        return np.transpose(arr, (1, 2, 0))
    if layout != 'auto':
        raise ValueError('output_layout must be one of: auto, hwc, chw')

    if arr.ndim != 3:
        return arr
    c_first, h, w = arr.shape
    if c_first <= 32 and c_first < h and c_first < w:
        return np.transpose(arr, (1, 2, 0))
    return arr


def _denormalize_from_minus1_1(img: np.ndarray, out_dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        maxv = float(info.max)
        mid = maxv / 2.0
        out = img * mid + mid
        out = np.clip(out, 0.0, maxv)
        return out.astype(out_dtype)
    return img.astype(out_dtype)


def _map_pan_name(ms_name: str) -> str:
    if not FLAGS.pan_name_replace:
        return ms_name
    parts = [p.strip() for p in FLAGS.pan_name_replace.split(',', 1)]
    if len(parts) != 2:
        raise ValueError('pan_name_replace must be "from,to" or empty.')
    src, dst = parts
    return ms_name.replace(src, dst, 1) if src else ms_name


def read_pair(pan_folder: str, ms_folder: str, fname: str, ratio: int):
    """Read a PAN/MS pair, normalize to [-1,1], and return NCHW ready for model."""
    pan_name = _map_pan_name(fname)
    pan_path = os.path.join(pan_folder, pan_name)
    ms_path = os.path.join(ms_folder, fname)

    pan_raw = _ensure_hwc(tifffile.imread(pan_path))
    ms_raw = _ensure_hwc(tifffile.imread(ms_path))

    # PAN: force 1 band
    if pan_raw.shape[2] != 1:
        pan_raw = pan_raw[:, :, 0:1]

    ms_dtype = ms_raw.dtype

    pan = _robust_percentile_normalize_to_minus1_1(pan_raw)
    ms = _robust_percentile_normalize_to_minus1_1(ms_raw)

    # If MS is still lower-res, upsample to PAN size for inference.
    if ms.shape[0] != pan.shape[0] or ms.shape[1] != pan.shape[1]:
        # Use bicubic like the original code.
        ms = cv2.resize(ms, (pan.shape[1], pan.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Add batch dimension
    pan = pan.reshape((1, pan.shape[0], pan.shape[1], 1)).astype(np.float32)
    ms = ms.reshape((1, ms.shape[0], ms.shape[1], ms.shape[2])).astype(np.float32)

    return pan, ms, ms_dtype


def main(_):
    if not os.path.exists(FLAGS.result_path):
        os.makedirs(FLAGS.result_path)

    # pan_size/ms_size are not used as fixed shapes anymore (the model is fully-conv),
    # but we keep the constructor signature for compatibility.
    model = PanGan(
        pan_size=None,
        ms_size=None,
        batch_size=FLAGS.batch_size,
        num_spectrum=FLAGS.num_spectrum,
        ratio=FLAGS.ratio,
        init_lr=0.001,
        lr_decay_rate=0.99,
        lr_decay_step=1000,
        is_training=False,
    )

    saver = tf.train.Saver(
        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Pan_model')
    )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, FLAGS.model_path)

        ms_folder = os.path.join(FLAGS.test_path, FLAGS.ms_dir)
        pan_folder = os.path.join(FLAGS.test_path, FLAGS.pan_dir)

        ms_files = {f for f in os.listdir(ms_folder) if f.lower().endswith(('.tif', '.tiff'))}
        pan_files = {f for f in os.listdir(pan_folder) if f.lower().endswith(('.tif', '.tiff'))}
        fnames = sorted([f for f in ms_files if _map_pan_name(f) in pan_files])
        if len(fnames) == 0:
            raise FileNotFoundError(
                f"No matching .tif/.tiff filenames between {ms_folder} and {pan_folder} "
                f"using pan_name_replace={FLAGS.pan_name_replace!r}"
            )

        for fname in fnames:
            print(fname)
            start = time.time()

            pan, ms, ms_dtype = read_pair(pan_folder, ms_folder, fname, FLAGS.ratio)

            out, err_spec, err_spa = sess.run(
                [model.PanSharpening_img, model.g_spectrum_loss, model.g_spatial_loss],
                feed_dict={model.pan_img: pan, model.ms_img: ms},
            )

            out = _ensure_output_hwc(out.squeeze())  # (H,W,C)

            # Decide output dtype
            if FLAGS.save_like_input_dtype:
                out_dtype = ms_dtype if np.issubdtype(ms_dtype, np.integer) else np.uint16
            else:
                out_dtype = np.uint8

            out_int = _denormalize_from_minus1_1(out, out_dtype)

            save_name = os.path.splitext(fname)[0] + '.tif'
            save_path = os.path.join(FLAGS.result_path, save_name)
            tifffile.imwrite(
                save_path,
                out_int,
                metadata={'axes': 'YXC'},
                photometric='minisblack',
                planarconfig='CONTIG',
            )

            print(
                f"{fname} done. time={time.time() - start:.3f}s | "
                f"spectrum_error={err_spec:.6f} | spatial_error={err_spa:.6f}"
            )


if __name__ == '__main__':
    tf.app.run()
