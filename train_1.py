import tensorflow as tf
import os
import time
import csv
import math
from typing import List, Dict, Any

# Optional: matplotlib is only used to save the loss curve image at the end of training.
# If it's not installed, we will still write CSV logs and the best-model list.
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None

import numpy as np
from PanGan import PanGan
from DataSet import DataSet
from config import FLAGES


def _to_float(x: Any) -> float:
    try:
        return float(np.asarray(x).squeeze())
    except Exception:
        try:
            return float(x)
        except Exception:
            return math.nan


def _ensure_dir(path: str) -> None:
    if path and (not os.path.exists(path)):
        os.makedirs(path, exist_ok=True)


def _save_loss_curve_png(
    steps: List[int],
    g_losses: List[float],
    d_spatial_losses: List[float],
    d_spectrum_losses: List[float],
    g_hp_losses: List[float],
    g_spec_losses: List[float],
    adv_weights: List[float],
    save_path: str,
    max_points: int = 20000,
) -> bool:
    if plt is None:
        print('[WARN] matplotlib is not available. Skip saving loss curve image:', save_path)
        return False

    n = len(steps)
    if n == 0:
        print('[WARN] No loss records found. Skip saving loss curve image:', save_path)
        return False

    # downsample for huge runs
    if n > max_points:
        stride = int(n / max_points) + 1
        idx = list(range(0, n, stride))
        steps = [steps[i] for i in idx]
        g_losses = [g_losses[i] for i in idx]
        d_spatial_losses = [d_spatial_losses[i] for i in idx]
        d_spectrum_losses = [d_spectrum_losses[i] for i in idx]
        g_hp_losses = [g_hp_losses[i] for i in idx]
        g_spec_losses = [g_spec_losses[i] for i in idx]
        adv_weights = [adv_weights[i] for i in idx]

    plt.figure(figsize=(12, 7))
    plt.plot(steps, g_losses, label='g_loss')
    plt.plot(steps, g_hp_losses, label='g_spatial_hp')
    plt.plot(steps, g_spec_losses, label='g_spectral')
    plt.plot(steps, d_spatial_losses, label='d_spatial_loss')
    plt.plot(steps, d_spectrum_losses, label='d_spectrum_loss')
    plt.plot(steps, adv_weights, label='adv_weight')
    plt.xlabel('Iteration')
    plt.ylabel('Loss / Weight')
    plt.title('Training Curves')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return True


def _rank_best_models(saved_models: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    filtered = [m for m in saved_models if (m.get('g_loss') is not None and not math.isnan(m['g_loss']))]
    filtered.sort(key=lambda d: d['g_loss'])
    return filtered[:k]


def main(argv):
    # ------------------------------
    # Build model
    # ------------------------------
    model = PanGan(
        pan_size=FLAGES.pan_size,
        ms_size=FLAGES.ms_size,
        batch_size=FLAGES.batch_size,
        num_spectrum=FLAGES.num_spectrum,
        ratio=FLAGES.ratio,
        init_lr=FLAGES.lr,
        lr_decay_rate=FLAGES.decay_rate,
        lr_decay_step=FLAGES.decay_step,
        is_training=True,

        # optional knobs (use getattr for backward compatibility)
        d_lr=getattr(FLAGES, 'd_lr', None),
        beta1=getattr(FLAGES, 'beta1', 0.5),
        beta2=getattr(FLAGES, 'beta2', 0.999),
        grad_clip_norm=getattr(FLAGES, 'grad_clip_norm', 5.0),

        lambda_hp=getattr(FLAGES, 'lambda_hp', 5.0),
        lambda_spec=getattr(FLAGES, 'lambda_spec', 1.0),
        lambda_adv_spatial=getattr(FLAGES, 'lambda_adv_spatial', 1.0),
        lambda_adv_spectrum=getattr(FLAGES, 'lambda_adv_spectrum', 1.0),

        adv_warmup_iters=getattr(FLAGES, 'adv_warmup_iters', 0),
        adv_ramp_iters=getattr(FLAGES, 'adv_ramp_iters', 8000),
        adv_weight_max=getattr(FLAGES, 'adv_weight_max', 1.0),

        residual_scale=getattr(FLAGES, 'residual_scale', 0.1),
        use_instance_norm=getattr(FLAGES, 'use_instance_norm', True),

        d_real_label=getattr(FLAGES, 'd_real_label', 0.9),
        d_fake_label=getattr(FLAGES, 'd_fake_label', 0.0),
    )
    model.train()

    # ------------------------------
    # Dataset
    # ------------------------------
    dataset = DataSet(
        FLAGES.pan_size,
        FLAGES.ms_size,
        FLAGES.img_path,
        FLAGES.data_path,
        FLAGES.batch_size,
        FLAGES.stride,
        category='train',
        norm_mode=getattr(FLAGES, 'norm_mode', 'percentile'),
        p_low=getattr(FLAGES, 'norm_p_low', 2.0),
        p_high=getattr(FLAGES, 'norm_p_high', 98.0),
        sample_pixels=getattr(FLAGES, 'norm_sample_pixels', 200000),
        augment=getattr(FLAGES, 'augment', True),
        aug_hflip=getattr(FLAGES, 'aug_hflip', True),
        aug_vflip=getattr(FLAGES, 'aug_vflip', True),
        aug_rot90=getattr(FLAGES, 'aug_rot90', True),
        noise_std=getattr(FLAGES, 'aug_noise_std', 0.0),
    )
    DataGenerator = dataset.data_generator

    dataset_valid = DataSet(
        FLAGES.pan_size,
        FLAGES.ms_size,
        FLAGES.img_path,
        FLAGES.data_path,
        FLAGES.batch_size,
        FLAGES.stride,
        category='valid',
        norm_mode=getattr(FLAGES, 'norm_mode', 'percentile'),
        p_low=getattr(FLAGES, 'norm_p_low', 2.0),
        p_high=getattr(FLAGES, 'norm_p_high', 98.0),
        sample_pixels=getattr(FLAGES, 'norm_sample_pixels', 200000),
        augment=False,
    )
    DataGenerator_valid = dataset_valid.data_generator

    merge_summary = tf.summary.merge_all()
    _ensure_dir(FLAGES.log_dir)
    _ensure_dir(FLAGES.model_save_dir)

    # ---- log/report settings ----
    loss_log_iters = int(getattr(FLAGES, 'loss_log_iters', 10))
    loss_plot_max_points = int(getattr(FLAGES, 'loss_plot_max_points', 20000))
    best_k = int(getattr(FLAGES, 'best_k', 5))
    flush_every = int(getattr(FLAGES, 'loss_log_flush_every', 200))
    d_steps = int(getattr(FLAGES, 'd_steps', 1))
    g_steps = int(getattr(FLAGES, 'g_steps', 1))

    run_tag = time.strftime('%Y%m%d-%H%M%S')
    report_dir = os.path.join(FLAGES.model_save_dir, 'training_report')
    _ensure_dir(report_dir)
    loss_csv_path = os.path.join(report_dir, f'loss_history_{run_tag}.csv')
    loss_png_path = os.path.join(report_dir, f'loss_curve_{run_tag}.png')
    best_txt_path = os.path.join(report_dir, f'best_models_{run_tag}.txt')

    print('[INFO] Loss CSV will be saved to:', loss_csv_path)
    print('[INFO] Loss curve image will be saved to:', loss_png_path)
    print('[INFO] Best-model list will be saved to:', best_txt_path)
    print('[INFO] d_steps={}, g_steps={}'.format(d_steps, g_steps))

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(FLAGES.log_dir, sess.graph)
        saver_all = tf.train.Saver(max_to_keep=None)
        saver_g = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Pan_model'), max_to_keep=None)

        sess.run(tf.global_variables_initializer())

        if getattr(FLAGES, 'is_pretrained', False):
            # NOTE: set your own checkpoint path if needed
            ckpt = getattr(FLAGES, 'pretrained_path', None)
            if ckpt:
                saver_all.restore(sess, ckpt)
                print('[INFO] Restored pretrained model:', ckpt)

        steps_hist: List[int] = []
        g_hist: List[float] = []
        d_spa_hist: List[float] = []
        d_spe_hist: List[float] = []
        g_hp_hist: List[float] = []
        g_spec_hist: List[float] = []
        adv_hist: List[float] = []
        saved_models: List[Dict[str, Any]] = []

        csv_rows_written = 0
        csv_f = open(loss_csv_path, 'w', newline='', encoding='utf-8')
        csv_w = csv.writer(csv_f)
        csv_w.writerow([
            'step',
            'g_loss',
            'g_spatial_hp',
            'g_spectral',
            'd_spatial_loss',
            'd_spectrum_loss',
            'adv_weight',
            'lr_g',
            'lr_d',
            'time_sec'
        ])

        def _finalize_reports():
            try:
                csv_f.flush()
                csv_f.close()
            except Exception:
                pass

            _save_loss_curve_png(
                steps=steps_hist,
                g_losses=g_hist,
                d_spatial_losses=d_spa_hist,
                d_spectrum_losses=d_spe_hist,
                g_hp_losses=g_hp_hist,
                g_spec_losses=g_spec_hist,
                adv_weights=adv_hist,
                save_path=loss_png_path,
                max_points=loss_plot_max_points,
            )

            best = _rank_best_models(saved_models, k=best_k)
            with open(best_txt_path, 'w', encoding='utf-8') as f:
                if not best:
                    f.write('No checkpoints saved.\n')
                else:
                    f.write(f'Top-{best_k} checkpoints by smallest g_loss:\n')
                    for rank, rec in enumerate(best, start=1):
                        f.write(
                            f"#{rank} step={rec.get('step')}  g_loss={rec.get('g_loss'):.6f}  "
                            f"Generator={rec.get('generator_ckpt')}  PanNet={rec.get('pannet_ckpt')}\n"
                        )

            print('\n================ Best checkpoints (by smallest g_loss) ================')
            if best:
                for rank, rec in enumerate(best, start=1):
                    print(f"#{rank} step={rec.get('step')} g_loss={rec.get('g_loss'):.6f}  {rec.get('generator_ckpt')}")
            else:
                print('[INFO] No checkpoints saved during this run.')
            print('======================================================================\n')

        try:
            for itr in range(int(FLAGES.iters)):
                t0 = time.time()
                pan_batch, ms_batch = next(DataGenerator)

                # ---- train D ----
                for _ in range(max(1, d_steps)):
                    sess.run(
                        model.train_spatial_discrim,
                        feed_dict={model.pan_img: pan_batch, model.ms_img: ms_batch},
                    )
                    sess.run(
                        model.train_spectrum_discrim,
                        feed_dict={model.pan_img: pan_batch, model.ms_img: ms_batch},
                    )

                # ---- train G ----
                for _ in range(max(1, g_steps)):
                    sess.run(
                        model.train_Pan_model,
                        feed_dict={model.pan_img: pan_batch, model.ms_img: ms_batch},
                    )

                # ---- fetch losses for logging ----
                fetch = [
                    model.global_step,
                    model.g_loss,
                    model.g_spatial_loss,
                    model.g_spectrum_loss,
                    model.spatial_loss,
                    model.spectrum_loss,
                    model.adv_weight if hasattr(model, 'adv_weight') else tf.constant(0.0),
                    model.learning_rate_g if hasattr(model, 'learning_rate_g') else tf.constant(0.0),
                    model.learning_rate_d if hasattr(model, 'learning_rate_d') else tf.constant(0.0),
                    merge_summary,
                ]
                gs, g_loss, g_hp, g_spec, d_spa, d_spe, adv_w, lr_g, lr_d, summary = sess.run(
                    fetch, feed_dict={model.pan_img: pan_batch, model.ms_img: ms_batch}
                )

                step = int(gs)
                time_sec = _to_float(time.time() - t0)

                # console print every loss_log_iters (avoid flooding for 50k)
                if loss_log_iters <= 1 or (step % loss_log_iters == 0):
                    print(
                        f"Step {step}/{FLAGES.iters} | "
                        f"g_loss={_to_float(g_loss):.6f} (hp={_to_float(g_hp):.6f}, spec={_to_float(g_spec):.6f}) | "
                        f"d_spa={_to_float(d_spa):.6f} d_spe={_to_float(d_spe):.6f} | "
                        f"adv_w={_to_float(adv_w):.3f} | lr_g={_to_float(lr_g):.7f} lr_d={_to_float(lr_d):.7f} | "
                        f"{time_sec:.3f}s"
                    )

                    train_writer.add_summary(summary, step)

                    # history arrays
                    steps_hist.append(step)
                    g_hist.append(_to_float(g_loss))
                    g_hp_hist.append(_to_float(g_hp))
                    g_spec_hist.append(_to_float(g_spec))
                    d_spa_hist.append(_to_float(d_spa))
                    d_spe_hist.append(_to_float(d_spe))
                    adv_hist.append(_to_float(adv_w))

                    csv_w.writerow([
                        step,
                        _to_float(g_loss),
                        _to_float(g_hp),
                        _to_float(g_spec),
                        _to_float(d_spa),
                        _to_float(d_spe),
                        _to_float(adv_w),
                        _to_float(lr_g),
                        _to_float(lr_d),
                        time_sec,
                    ])
                    csv_rows_written += 1
                    if flush_every > 0 and (csv_rows_written % flush_every == 0):
                        csv_f.flush()

                # checkpoints
                if step % int(FLAGES.model_save_iters) == 0:
                    pan_ckpt = saver_all.save(
                        sess=sess,
                        save_path=os.path.join(FLAGES.model_save_dir, 'PanNet'),
                        global_step=step,
                    )
                    gen_ckpt = saver_g.save(
                        sess=sess,
                        save_path=os.path.join(FLAGES.model_save_dir, 'Generator'),
                        global_step=step,
                    )
                    saved_models.append(
                        {
                            'step': step,
                            'g_loss': _to_float(g_loss),
                            'generator_ckpt': os.path.basename(gen_ckpt),
                            'pannet_ckpt': os.path.basename(pan_ckpt),
                        }
                    )
                    print('\n[INFO] Model checkpoint saved (step={})\n'.format(step))

                if step >= int(FLAGES.iters):
                    break

        except KeyboardInterrupt:
            print('\n[INFO] Training interrupted by user (KeyboardInterrupt). Generating report...')
        finally:
            _finalize_reports()

        print('Training done.')


if __name__ == '__main__':
    tf.app.run()
