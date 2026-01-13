import tensorflow as tf
import os
import time
import csv
import math
from typing import List, Tuple, Dict, Any

# Optional: matplotlib is only used to save the loss curve image at the end of training.
# If it's not installed, we will still write CSV logs and the best-model list.
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
except Exception:
    plt = None

import numpy as np
from PanGan import PanGan
from DataSet import DataSet
from config import FLAGES

def print_current_training_stats(error_pan_model, error_ms_model, error_g_model, global_step, learning_rate, time_elapsed):
    stats = 'Step: {}/{} ----- Cur_lr: {:1.7f} ----- Time: {:>2.2f} sec.'.format(global_step, FLAGES.iters,
                                                                                 learning_rate, time_elapsed)
    losses =  ' | spatial loss: {}'.format(error_pan_model)
    losses += ' | spectrual loss: {}'.format(error_ms_model)
    losses += ' | generator loss: {}'.format(error_g_model)
    print(stats)
    print(losses + '\n')
    
def print_current_training_stats_valid(error_spatial, error_spectrual, global_step, learning_rate, time_elapsed):
    stats = 'Valid_Step: {}/{} ----- Cur_lr: {:1.7f} ----- Time: {:>2.2f} sec.'.format(global_step, FLAGES.iters,
                                                                                 learning_rate, time_elapsed)
    losses =  ' | spatial error: {}'.format(error_spatial)
    losses += ' | spectrual error: {}'.format(error_spectrual)
    print(stats)
    print(losses + '\n')


def _to_float(x: Any) -> float:
    """Convert numpy scalar / Python scalar to float safely."""
    try:
        return float(np.asarray(x).squeeze())
    except Exception:
        try:
            return float(x)
        except Exception:
            return math.nan


def _ensure_dir(path: str) -> None:
    if path and (not os.path.exists(path)):
        os.makedirs(path)


def _save_loss_curve_png(
    steps: List[int],
    spatial_losses: List[float],
    spectrum_losses: List[float],
    g_losses: List[float],
    save_path: str,
    max_points: int = 20000,
) -> bool:
    """Save a loss curve png. Returns True if saved, False otherwise."""
    if plt is None:
        print('[WARN] matplotlib is not available. Skip saving loss curve image:', save_path)
        return False

    n = len(steps)
    if n == 0:
        print('[WARN] No loss records found. Skip saving loss curve image:', save_path)
        return False

    # Downsample to avoid very slow plotting for huge runs.
    if n > max_points:
        stride = int(n / max_points) + 1
        idx = list(range(0, n, stride))
        steps_p = [steps[i] for i in idx]
        spatial_p = [spatial_losses[i] for i in idx]
        spectrum_p = [spectrum_losses[i] for i in idx]
        g_p = [g_losses[i] for i in idx]
    else:
        steps_p, spatial_p, spectrum_p, g_p = steps, spatial_losses, spectrum_losses, g_losses

    plt.figure(figsize=(11, 6))
    plt.plot(steps_p, g_p, label='g_loss')
    plt.plot(steps_p, spatial_p, label='spatial_loss')
    plt.plot(steps_p, spectrum_p, label='spectrum_loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return True


def _rank_best_models(
    saved_models: List[Dict[str, Any]],
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Rank checkpoints by generator loss (ascending)."""
    # Filter out nan
    filtered = [m for m in saved_models if (m.get('g_loss') is not None and not math.isnan(m['g_loss']))]
    filtered.sort(key=lambda d: d['g_loss'])
    return filtered[:k]

def main(argv):
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
        d_lr=getattr(FLAGES, 'd_lr', None),
        lambda_hp=getattr(FLAGES, 'lambda_hp', 5.0),
        lambda_spec=getattr(FLAGES, 'lambda_spec', 1.0),
        lambda_adv_spatial=getattr(FLAGES, 'lambda_adv_spatial', 1.0),
        lambda_adv_spectrum=getattr(FLAGES, 'lambda_adv_spectrum', 1.0),
        adv_warmup_iters=getattr(FLAGES, 'adv_warmup_iters', 2000),
        adv_ramp_iters=getattr(FLAGES, 'adv_ramp_iters', 8000),
        adv_weight_max=getattr(FLAGES, 'adv_weight_max', 1.0),
        residual_scale=getattr(FLAGES, 'residual_scale', 0.1),
        beta1=getattr(FLAGES, 'beta1', 0.5),
        beta2=getattr(FLAGES, 'beta2', 0.999),
        grad_clip_norm=getattr(FLAGES, 'grad_clip_norm', 5.0),
    )
    model.train()
    dataset=DataSet(FLAGES.pan_size, FLAGES.ms_size, FLAGES.img_path, FLAGES.data_path, FLAGES.batch_size,
                    FLAGES.stride)
    DataGenerator=dataset.data_generator
    
    dataset_valid=DataSet(FLAGES.pan_size, FLAGES.ms_size, FLAGES.img_path, FLAGES.data_path, FLAGES.batch_size,
                    FLAGES.stride, 'valid')
    DataGenerator_valid=dataset_valid.data_generator

    merge_summary=tf.summary.merge_all()
    if not os.path.exists(FLAGES.log_dir):
        os.makedirs(FLAGES.log_dir)
    if not os.path.exists(FLAGES.model_save_dir):
        os.makedirs(FLAGES.model_save_dir)

    # ---- Loss logging / report settings (all optional in config.py) ----
    loss_log_iters = int(getattr(FLAGES, 'loss_log_iters', 1))           # record losses every N iterations
    loss_plot_max_points = int(getattr(FLAGES, 'loss_plot_max_points', 20000))
    best_k = int(getattr(FLAGES, 'best_k', 5))
    flush_every = int(getattr(FLAGES, 'loss_log_flush_every', 200))      # flush CSV every N written rows

    run_tag = time.strftime('%Y%m%d-%H%M%S')
    report_dir = os.path.join(FLAGES.model_save_dir, 'training_report')
    _ensure_dir(report_dir)
    loss_csv_path = os.path.join(report_dir, f'loss_history_{run_tag}.csv')
    loss_png_path = os.path.join(report_dir, f'loss_curve_{run_tag}.png')
    best_txt_path = os.path.join(report_dir, f'best_models_{run_tag}.txt')

    print('[INFO] Loss CSV will be saved to:', loss_csv_path)
    print('[INFO] Loss curve image will be saved to:', loss_png_path)
    print('[INFO] Best-model list will be saved to:', best_txt_path)

    with tf.Session() as sess:
        train_writer=tf.summary.FileWriter(FLAGES.log_dir, sess.graph)
        saver=tf.train.Saver(max_to_keep=None)
        saver_g=tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Pan_model'),max_to_keep=None)
        init_op=tf.global_variables_initializer()
        sess.run(init_op)

        if FLAGES.is_pretrained:
            saver.restore(sess, './model/qk/PanNet-107000')

        # In-memory history for plotting (downsampled for huge runs) + CSV for full log.
        steps_hist: List[int] = []
        spatial_hist: List[float] = []
        spectrum_hist: List[float] = []
        g_hist: List[float] = []
        saved_models: List[Dict[str, Any]] = []

        csv_rows_written = 0
        csv_f = open(loss_csv_path, 'w', newline='', encoding='utf-8')
        csv_w = csv.writer(csv_f)
        csv_w.writerow(['step', 'spatial_loss', 'spectrum_loss', 'g_loss', 'learning_rate', 'time_sec'])

        def _finalize_reports():
            """Write loss curve + best models even if training is interrupted."""
            try:
                csv_f.flush()
            except Exception:
                pass
            try:
                csv_f.close()
            except Exception:
                pass

            # Save plot
            _save_loss_curve_png(
                steps=steps_hist,
                spatial_losses=spatial_hist,
                spectrum_losses=spectrum_hist,
                g_losses=g_hist,
                save_path=loss_png_path,
                max_points=loss_plot_max_points,
            )

            # Rank and save best models
            best = _rank_best_models(saved_models, k=best_k)
            with open(best_txt_path, 'w', encoding='utf-8') as f:
                if not best:
                    f.write('No model checkpoints were saved, so best-model ranking is empty.\n')
                else:
                    f.write(f'Top-{best_k} checkpoints by smallest g_loss (ascending):\n')
                    for rank, rec in enumerate(best, start=1):
                        f.write(
                            f"#{rank}  step={rec.get('step')}  g_loss={rec.get('g_loss'):.6f}  "
                            f"spatial_loss={rec.get('spatial_loss'):.6f}  spectrum_loss={rec.get('spectrum_loss'):.6f}  "
                            f"Generator={rec.get('generator_ckpt')}  PanNet={rec.get('pannet_ckpt')}\n"
                        )

            # Also print to console for convenience
            print('\n================ Best checkpoints (by smallest g_loss) ================')
            if not best:
                print('[INFO] No checkpoints saved during this run.')
            else:
                for rank, rec in enumerate(best, start=1):
                    print(
                        f"#{rank} step={rec.get('step')}  g_loss={rec.get('g_loss'):.6f}  "
                        f"Generator={rec.get('generator_ckpt')}"
                    )
            print('======================================================================\n')

        try:
            for training_itr in range(FLAGES.iters):
                t1 = time.time()
                pan_batch, ms_batch = next(DataGenerator)

                for i in range(2):
                    _, error_pan_model = sess.run(
                        [model.train_spatial_discrim, model.spatial_loss],
                        feed_dict={model.pan_img: pan_batch, model.ms_img: ms_batch},
                    )
                    _, error_ms_model = sess.run(
                        [model.train_spectrum_discrim, model.spectrum_loss],
                        feed_dict={model.pan_img: pan_batch, model.ms_img: ms_batch},
                    )

                _, error_g_model, global_step, summary, learning_rate = sess.run(
                    [model.train_Pan_model, model.g_loss, model.global_step, merge_summary, model.learning_rate],
                    feed_dict={model.pan_img: pan_batch, model.ms_img: ms_batch},
                )

                # Re-evaluate D losses after G update (matches original code behavior).
                error_pan_model = sess.run(
                    model.spatial_loss, feed_dict={model.pan_img: pan_batch, model.ms_img: ms_batch}
                )
                error_ms_model = sess.run(
                    model.spectrum_loss, feed_dict={model.pan_img: pan_batch, model.ms_img: ms_batch}
                )

                # Convert to python floats for logging.
                g_loss_f = _to_float(error_g_model)
                spatial_f = _to_float(error_pan_model)
                spectrum_f = _to_float(error_ms_model)
                lr_f = _to_float(learning_rate)
                time_sec = _to_float(time.time() - t1)

                # Use checkpoint step convention (same as saver.save(..., global_step=global_step+1)).
                try:
                    step = int(global_step) + 1
                except Exception:
                    step = int(training_itr) + 1

                print_current_training_stats(spatial_f, spectrum_f, g_loss_f, step, lr_f, time_sec)
                train_writer.add_summary(summary, step)

                # Record loss history
                if loss_log_iters <= 1 or (step % loss_log_iters == 0):
                    steps_hist.append(step)
                    spatial_hist.append(spatial_f)
                    spectrum_hist.append(spectrum_f)
                    g_hist.append(g_loss_f)

                    csv_w.writerow([step, spatial_f, spectrum_f, g_loss_f, lr_f, time_sec])
                    csv_rows_written += 1
                    if flush_every > 0 and (csv_rows_written % flush_every == 0):
                        csv_f.flush()

                # Save checkpoints + record their loss
                if (step % FLAGES.model_save_iters) == 0:
                    pan_ckpt = saver.save(
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
                            'g_loss': g_loss_f,
                            'spatial_loss': spatial_f,
                            'spectrum_loss': spectrum_f,
                            'generator_ckpt': os.path.basename(gen_ckpt),
                            'pannet_ckpt': os.path.basename(pan_ckpt),
                            'generator_ckpt_path': gen_ckpt,
                            'pannet_ckpt_path': pan_ckpt,
                        }
                    )
                    print('\nModel checkpoint saved...  (step={})\n'.format(step))

                if step >= FLAGES.iters:
                    break

        except KeyboardInterrupt:
            print('\n[INFO] Training interrupted by user (KeyboardInterrupt). Generating report...')
        finally:
            _finalize_reports()

        print('Training done.')

if __name__ == '__main__':
    tf.app.run()


