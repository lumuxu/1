import os
import glob
import numpy as np
import h5py
import tifffile


def _list_tif_files(folder: str):
    exts = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)


def _ensure_hwc(arr: np.ndarray) -> np.ndarray:
    """Ensure image array is (H, W, C)."""
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported TIFF shape: {arr.shape}")

    # Many remote sensing TIFFs are stored as (C, H, W)
    c_first, h, w = arr.shape
    if c_first <= 32 and c_first < h and c_first < w:
        return np.transpose(arr, (1, 2, 0))
    return arr


def _robust_percentile_normalize_to_minus1_1(img: np.ndarray, p_low=0.2, p_high=99.8) -> np.ndarray:
    """Robustly normalize to [-1, 1] using percentiles.

    Why:
    - Remote-sensing uint16 TIFF often uses only a small portion of [0,65535].
      If we normalize by the full dtype range, almost all values collapse near -1.
    - Percentile scaling keeps contrast & gradients meaningful and tends to make GAN converge.

    Behavior:
    - For integer inputs: per-channel percentile scaling.
    - For float inputs:
        * if values look like [0,1] -> map to [-1,1]
        * else still apply percentile scaling (safer when range is large)
    """
    img_f = img.astype(np.float32)

    # special case: [0,1] floats
    if np.issubdtype(img.dtype, np.floating):
        vmin = float(np.nanmin(img_f))
        vmax = float(np.nanmax(img_f))
        if vmin >= 0.0 and vmax <= 1.0:
            return img_f * 2.0 - 1.0

    # Percentile scaling (per-channel)
    if img_f.ndim == 2:
        img_f = img_f[:, :, None]

    out = np.empty_like(img_f, dtype=np.float32)
    for c in range(img_f.shape[2]):
        ch = img_f[:, :, c]
        lo = np.nanpercentile(ch, p_low)
        hi = np.nanpercentile(ch, p_high)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
            # fallback: use min/max
            lo = float(np.nanmin(ch))
            hi = float(np.nanmax(ch))
            if hi <= lo + 1e-6:
                out[:, :, c] = 0.0
                continue
        ch_n = (ch - lo) / (hi - lo)
        ch_n = np.clip(ch_n, 0.0, 1.0)
        out[:, :, c] = ch_n * 2.0 - 1.0

    return out


def _augment_pair(pan_patch: np.ndarray, ms_patch: np.ndarray) -> tuple:
    """Apply the same random geometric augmentation to PAN & MS patches."""
    # rotate 0/90/180/270
    k = np.random.randint(0, 4)
    if k:
        pan_patch = np.rot90(pan_patch, k, axes=(0, 1))
        ms_patch = np.rot90(ms_patch, k, axes=(0, 1))

    # flip H
    if np.random.rand() < 0.5:
        pan_patch = np.flip(pan_patch, axis=1)
        ms_patch = np.flip(ms_patch, axis=1)

    # flip V
    if np.random.rand() < 0.5:
        pan_patch = np.flip(pan_patch, axis=0)
        ms_patch = np.flip(ms_patch, axis=0)

    return pan_patch, ms_patch


class DataSet(object):
    """Patch dataset builder/loader for PanGAN.

    Expected folder structure:

        source_path/
            MS/   (multi-spectral .tif/.tiff)
            Pan/  (panchromatic .tif/.tiff)

    Files are paired by sorted order.

    Cached patches are stored in a single .h5 file at `data_save_path`.
    If the file already exists, patch extraction is skipped.

    IMPORTANT:
    - If you change normalization/augmentation logic, delete the existing .h5 cache
      or change `data_save_path`, otherwise old cached patches will be reused.
    """

    def __init__(self, pan_size, ms_size, source_path, data_save_path, batch_size, stride, category='train'):
        self.pan_size = int(pan_size)
        self.ms_size = int(ms_size)
        self.batch_size = int(batch_size)
        self.stride = int(stride)

        if not os.path.exists(data_save_path):
            parent = os.path.dirname(data_save_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self.make_data(source_path, data_save_path)

        self.pan, self.ms = self.read_data(data_save_path, category)
        self.data_generator = self.generator()

    # ------------------------------
    # Public generator (with on-the-fly augmentation)
    # ------------------------------
    def generator(self):
        num_data = self.pan.shape[0]
        pan_shape = self.pan.shape[1:]
        ms_shape = self.ms.shape[1:]

        while True:
            batch_pan = np.empty((self.batch_size,) + pan_shape, dtype=np.float32)
            batch_ms = np.empty((self.batch_size,) + ms_shape, dtype=np.float32)
            for i in range(self.batch_size):
                idx = np.random.randint(0, num_data)
                pan_p = self.pan[idx]
                ms_p = self.ms[idx]

                # Random augmentation (very helpful when you only have 1~2 images)
                pan_p, ms_p = _augment_pair(pan_p, ms_p)

                batch_pan[i] = pan_p
                batch_ms[i] = ms_p
            yield batch_pan, batch_ms

    # ------------------------------
    # H5 IO
    # ------------------------------
    @staticmethod
    def read_data(path, category):
        with h5py.File(path, 'r') as f:
            if category == 'train':
                pan = np.array(f['pan_train'])
                ms = np.array(f['ms_train'])
            else:
                pan = np.array(f['pan_valid'])
                ms = np.array(f['ms_valid'])
        return pan, ms

    # ------------------------------
    # Dataset building from TIFF
    # ------------------------------
    def make_data(self, source_path, data_save_path):
        ms_dir = os.path.join(source_path, 'MS')
        pan_dir = os.path.join(source_path, 'Pan')

        if not os.path.isdir(ms_dir) or not os.path.isdir(pan_dir):
            raise FileNotFoundError(
                "Cannot find MS/Pan subfolders. Expected: "
                f"{ms_dir} and {pan_dir}"
            )

        ms_files = _list_tif_files(ms_dir)
        pan_files = _list_tif_files(pan_dir)
        if len(ms_files) == 0:
            raise FileNotFoundError(f"No TIFF found in {ms_dir}")
        if len(pan_files) == 0:
            raise FileNotFoundError(f"No TIFF found in {pan_dir}")
        if len(ms_files) != len(pan_files):
            raise ValueError(
                f"MS/Pan file counts differ: {len(ms_files)} vs {len(pan_files)}. "
                "Please ensure they are paired (same count), or keep only one pair."
            )

        all_pan = []
        all_ms = []

        for ms_path, pan_path in zip(ms_files, pan_files):
            ms_img = self._read_tif(ms_path, expect_ms=True)
            pan_img = self._read_tif(pan_path, expect_ms=False)

            pan_patches, ms_patches = self._crop_pair_to_patches(pan_img, ms_img)

            all_pan.extend(pan_patches)
            all_ms.extend(ms_patches)

        if len(all_pan) != len(all_ms):
            raise RuntimeError(f"Patch count mismatch: pan={len(all_pan)} ms={len(all_ms)}")

        print('The number of ms patch is: ' + str(len(all_ms)))
        print('The number of pan patch is: ' + str(len(all_pan)))

        pan_train, pan_valid, ms_train, ms_valid = self.split_data(all_pan, all_ms)

        print('The number of pan_train patch is: ' + str(len(pan_train)))
        print('The number of pan_valid patch is: ' + str(len(pan_valid)))
        print('The number of ms_train patch is: ' + str(len(ms_train)))
        print('The number of ms_valid patch is: ' + str(len(ms_valid)))

        pan_train = np.asarray(pan_train, dtype=np.float32)
        pan_valid = np.asarray(pan_valid, dtype=np.float32)
        ms_train = np.asarray(ms_train, dtype=np.float32)
        ms_valid = np.asarray(ms_valid, dtype=np.float32)

        with h5py.File(data_save_path, 'w') as f:
            f.create_dataset('pan_train', data=pan_train)
            f.create_dataset('pan_valid', data=pan_valid)
            f.create_dataset('ms_train', data=ms_train)
            f.create_dataset('ms_valid', data=ms_valid)

    def _read_tif(self, path, expect_ms: bool) -> np.ndarray:
        try:
            img = tifffile.memmap(path)
        except Exception:
            img = tifffile.imread(path)
        img = _ensure_hwc(img)

        if not expect_ms:
            if img.shape[2] != 1:
                img = img[:, :, 0:1]
        return img

    # ------------------------------
    # Patch extraction
    # ------------------------------
    def _crop_pair_to_patches(self, pan_img: np.ndarray, ms_img: np.ndarray):
        """Crop one aligned PAN/MS pair into patch lists."""
        # Normalize (robust percentiles)
        pan_n = _robust_percentile_normalize_to_minus1_1(pan_img)
        ms_n = _robust_percentile_normalize_to_minus1_1(ms_img)

        # PAN patch size
        pan_h, pan_w, _ = pan_n.shape
        ms_h, ms_w, ms_c = ms_n.shape

        # If MS is lower-res, we crop MS patches of ms_size; PAN patches of pan_size.
        # Assumes MS and PAN are aligned and cover same area.

        pan_patches = []
        ms_patches = []

        # define sliding window positions for PAN patches
        for y in range(0, pan_h - self.pan_size + 1, self.stride):
            for x in range(0, pan_w - self.pan_size + 1, self.stride):
                pan_patch = pan_n[y:y + self.pan_size, x:x + self.pan_size, :]

                # Map to MS coordinates by ratio of sizes
                # If your MS already upsampled to PAN size, pan_size==ms_size and this becomes 1:1.
                y_ms = int(round(y * (ms_h / float(pan_h))))
                x_ms = int(round(x * (ms_w / float(pan_w))))

                # ensure within bounds
                y_ms = min(max(y_ms, 0), max(ms_h - self.ms_size, 0))
                x_ms = min(max(x_ms, 0), max(ms_w - self.ms_size, 0))

                ms_patch = ms_n[y_ms:y_ms + self.ms_size, x_ms:x_ms + self.ms_size, :]

                if ms_patch.shape[0] != self.ms_size or ms_patch.shape[1] != self.ms_size:
                    continue

                pan_patches.append(pan_patch.astype(np.float32))
                ms_patches.append(ms_patch.astype(np.float32))

        return pan_patches, ms_patches

    @staticmethod
    def split_data(pan, ms, valid_ratio=0.1, seed=1234):
        """Split patch list into train/valid."""
        assert len(pan) == len(ms)
        n = len(pan)
        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)

        n_valid = int(round(n * valid_ratio))
        valid_idx = idx[:n_valid]
        train_idx = idx[n_valid:]

        pan_train = [pan[i] for i in train_idx]
        ms_train = [ms[i] for i in train_idx]
        pan_valid = [pan[i] for i in valid_idx]
        ms_valid = [ms[i] for i in valid_idx]

        return pan_train, pan_valid, ms_train, ms_valid
