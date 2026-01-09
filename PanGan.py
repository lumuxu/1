import tensorflow as tf
import numpy as np


class PanGan(object):
    """PanGAN (multi-source remote-sensing image fusion / pan-sharpening) - stabilized TF1.15.

    Goals:
    - make training **actually converge** on tiny datasets (e.g. only 2 image pairs)
    - keep the original project API so `train.py/test2.py` can run unchanged

    Main stability changes:
    1) **Residual generator**:  fused = upsample(MS) + residual
       This makes spectral loss easy to minimize and avoids output drift.
    2) Spatial loss uses **high-pass (Laplacian)** and is **standardized** to remove scale mismatch.
    3) **Adversarial warmup + ramp**: first train only recon losses, then gradually add GAN.
    4) Adam + optional grad clipping (safer than RMSProp for GAN on small batches).

    Inputs are expected in [-1, 1].
    """

    def __init__(
        self,
        pan_size,
        ms_size,
        batch_size,
        num_spectrum,
        ratio,
        init_lr=1e-4,
        lr_decay_rate=0.99,
        lr_decay_step=10000,
        is_training=True,
        # discriminator lr (TTUR)
        d_lr=None,
        # losses
        lambda_hp=5.0,
        lambda_spec=1.0,
        lambda_adv_spatial=1.0,
        lambda_adv_spectrum=1.0,
        # adversarial schedule
        adv_warmup_iters=2000,
        adv_ramp_iters=8000,
        adv_weight_max=1.0,
        # residual scale
        residual_scale=0.1,
        # optim
        beta1=0.5,
        beta2=0.999,
        grad_clip_norm=5.0,
    ):
        self.num_spectrum = int(num_spectrum)
        self.is_training = bool(is_training)
        self.ratio = ratio
        self.batch_size = int(batch_size)
        self.pan_size = pan_size
        self.ms_size = ms_size

        self.init_lr = float(init_lr)
        self.d_lr = float(d_lr) if d_lr is not None else float(init_lr)
        self.lr_decay_rate = float(lr_decay_rate)
        self.lr_decay_step = int(lr_decay_step)

        self.lambda_hp = float(lambda_hp)
        self.lambda_spec = float(lambda_spec)
        self.lambda_adv_spatial = float(lambda_adv_spatial)
        self.lambda_adv_spectrum = float(lambda_adv_spectrum)

        self.adv_warmup_iters = int(adv_warmup_iters)
        self.adv_ramp_iters = int(adv_ramp_iters)
        self.adv_weight_max = float(adv_weight_max)

        self.residual_scale = float(residual_scale)

        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.grad_clip_norm = float(grad_clip_norm)

        # global step
        self.global_step = tf.contrib.framework.get_or_create_global_step()

        self.build_model(batch_size=self.batch_size, num_spectrum=self.num_spectrum, is_training=self.is_training)

    # ------------------------------------------------------------------
    # Build graph
    # ------------------------------------------------------------------
    def build_model(self, batch_size, num_spectrum, is_training):
        with tf.name_scope('input'):
            self.pan_img = tf.placeholder(tf.float32, shape=(batch_size, None, None, 1), name='pan_placeholder')
            self.ms_img = tf.placeholder(tf.float32, shape=(batch_size, None, None, num_spectrum), name='ms_placeholder')

            # MS resized to PAN resolution (bicubic)
            self.ms_img_ = self._resize_like(self.ms_img, self.pan_img, method=tf.image.ResizeMethod.BICUBIC)

        with tf.name_scope('PanSharpening'):
            self.PanSharpening_img = self.PanSharpening_model_residual(self.pan_img, self.ms_img)
            # pseudo-pan
            self.PanSharpening_img_pan = tf.reduce_mean(self.PanSharpening_img, axis=3, keepdims=True)

        # high-pass (Laplacian)
        self.pan_img_hp = self.high_pass_laplacian(self.pan_img)
        self.fake_hp = self.high_pass_laplacian(self.PanSharpening_img_pan)

        # standardized HP -> avoid scale mismatch
        pan_hp_n = self._standardize_feature(self.pan_img_hp)
        fake_hp_n = self._standardize_feature(self.fake_hp)

        # recon losses
        with tf.name_scope('recon_losses'):
            self.g_spatial_loss = tf.reduce_mean(tf.square(fake_hp_n - pan_hp_n))
            tf.summary.scalar('g_spatial_loss', self.g_spatial_loss)

            # L1 tends to converge better than L2 for spectral recon
            self.g_spectrum_loss = tf.reduce_mean(tf.abs(self.PanSharpening_img - self.ms_img_))
            tf.summary.scalar('g_spectrum_loss', self.g_spectrum_loss)

        if not is_training:
            # for test2.py
            self.g_loss = self.lambda_hp * self.g_spatial_loss + self.lambda_spec * self.g_spectrum_loss
            return

        # ------------------------------------------------------------------
        # Discriminators
        # ------------------------------------------------------------------
        with tf.name_scope('discriminators'):
            # Spatial D on HP features (1ch)
            d_spa_real = self.spatial_discriminator(pan_hp_n, reuse=False)
            d_spa_fake = self.spatial_discriminator(fake_hp_n, reuse=True)

            # Spectrum D on MS (Cch)
            d_spe_real = self.spectrum_discriminator(self.ms_img_, reuse=False)
            d_spe_fake = self.spectrum_discriminator(self.PanSharpening_img, reuse=True)

        # LSGAN
        with tf.name_scope('d_loss'):
            self.spatial_loss = tf.reduce_mean(tf.square(d_spa_real - 1.0)) + tf.reduce_mean(tf.square(d_spa_fake - 0.0))
            self.spectrum_loss = tf.reduce_mean(tf.square(d_spe_real - 1.0)) + tf.reduce_mean(tf.square(d_spe_fake - 0.0))
            tf.summary.scalar('spatial_loss', self.spatial_loss)
            tf.summary.scalar('spectrum_loss', self.spectrum_loss)

        with tf.name_scope('g_adv'):
            self.spatial_loss_ad = tf.reduce_mean(tf.square(d_spa_fake - 1.0))
            self.spectrum_loss_ad = tf.reduce_mean(tf.square(d_spe_fake - 1.0))
            tf.summary.scalar('spatial_loss_ad', self.spatial_loss_ad)
            tf.summary.scalar('spectrum_loss_ad', self.spectrum_loss_ad)

            self.adv_weight = self._adv_weight_schedule(self.global_step)
            tf.summary.scalar('adv_weight', self.adv_weight)

        # total g loss
        with tf.name_scope('g_loss'):
            recon = self.lambda_hp * self.g_spatial_loss + self.lambda_spec * self.g_spectrum_loss
            adv = (self.lambda_adv_spatial * self.spatial_loss_ad + self.lambda_adv_spectrum * self.spectrum_loss_ad)
            self.g_loss = recon + self.adv_weight * adv
            tf.summary.scalar('g_loss', self.g_loss)

    # ------------------------------------------------------------------
    # Train ops
    # ------------------------------------------------------------------
    def train(self):
        """Create training ops.

        Exposes the same attributes expected by `train.py`:
        - train_Pan_model
        - train_spatial_discrim
        - train_spectrum_discrim
        - learning_rate   (for logging; generator lr)
        """
        t_vars = tf.trainable_variables()
        g_vars = [v for v in t_vars if 'Pan_model' in v.name]
        d_spa_vars = [v for v in t_vars if 'spatial_discriminator' in v.name]
        d_spe_vars = [v for v in t_vars if 'spectrum_discriminator' in v.name]

        with tf.name_scope('train_step'):
            self.learning_rate_g = tf.train.exponential_decay(
                self.init_lr,
                global_step=self.global_step,
                decay_rate=self.lr_decay_rate,
                decay_steps=self.lr_decay_step,
                staircase=False,
            )
            self.learning_rate_d = tf.train.exponential_decay(
                self.d_lr,
                global_step=self.global_step,
                decay_rate=self.lr_decay_rate,
                decay_steps=self.lr_decay_step,
                staircase=False,
            )
            # ---- IMPORTANT: keep compatibility with train.py ----
            self.learning_rate = self.learning_rate_g

            tf.summary.scalar('learning_rate_g', self.learning_rate_g)
            tf.summary.scalar('learning_rate_d', self.learning_rate_d)

            opt_g = tf.train.AdamOptimizer(self.learning_rate_g, beta1=self.beta1, beta2=self.beta2)
            opt_d = tf.train.AdamOptimizer(self.learning_rate_d, beta1=self.beta1, beta2=self.beta2)

            # G gradients
            g_grads_and_vars = opt_g.compute_gradients(self.g_loss, var_list=g_vars)
            g_grads_and_vars = self._clip_grads(g_grads_and_vars, name='g')
            self.train_Pan_model = opt_g.apply_gradients(g_grads_and_vars, global_step=self.global_step)

            # D gradients (no global_step increment)
            spa_grads_and_vars = opt_d.compute_gradients(self.spatial_loss, var_list=d_spa_vars)
            spe_grads_and_vars = opt_d.compute_gradients(self.spectrum_loss, var_list=d_spe_vars)
            spa_grads_and_vars = self._clip_grads(spa_grads_and_vars, name='d_spatial')
            spe_grads_and_vars = self._clip_grads(spe_grads_and_vars, name='d_spectrum')

            self.train_spatial_discrim = opt_d.apply_gradients(spa_grads_and_vars)
            self.train_spectrum_discrim = opt_d.apply_gradients(spe_grads_and_vars)

    # ------------------------------------------------------------------
    # Generator (residual)
    # ------------------------------------------------------------------
    def PanSharpening_model_residual(self, pan_img, ms_img):
        """Residual generator: fused = upsample(MS) + residual"""
        with tf.variable_scope('Pan_model', reuse=tf.AUTO_REUSE):
            ms_up = self._resize_like(ms_img, pan_img, method=tf.image.ResizeMethod.BICUBIC)
            x = tf.concat([ms_up, pan_img], axis=-1)

            x = self._conv(x, 64, k=7, s=1, name='c1')
            x = self.lrelu(x)
            x = self._conv(x, 32, k=5, s=1, name='c2')
            x = self.lrelu(x)
            res = self._conv(x, self.num_spectrum, k=3, s=1, name='c3')

            res = tf.tanh(res) * self.residual_scale
            fused = ms_up + res
            fused = tf.clip_by_value(fused, -1.0, 1.0)
            return fused

    # ------------------------------------------------------------------
    # Discriminators
    # ------------------------------------------------------------------
    def spatial_discriminator(self, img, reuse=False):
        """Spatial discriminator on 1-channel HP feature."""
        with tf.variable_scope('spatial_discriminator', reuse=reuse):
            x = self._conv(img, 32, k=3, s=2, name='c1')
            x = self.lrelu(x)
            x = self._conv(x, 64, k=3, s=2, name='c2')
            x = self.lrelu(self._inorm(x, 'in2'))
            x = self._conv(x, 128, k=3, s=2, name='c3')
            x = self.lrelu(self._inorm(x, 'in3'))
            x = self._conv(x, 256, k=3, s=2, name='c4')
            x = self.lrelu(self._inorm(x, 'in4'))
            # global average pooling -> [B,256]
            gap = tf.reduce_mean(x, axis=[1, 2])
            out = tf.layers.dense(gap, 1, name='fc')
            return out

    def spectrum_discriminator(self, img, reuse=False):
        """Spectral discriminator on multi-band MS."""
        with tf.variable_scope('spectrum_discriminator', reuse=reuse):
            x = self._conv(img, 32, k=3, s=2, name='c1')
            x = self.lrelu(x)
            x = self._conv(x, 64, k=3, s=2, name='c2')
            x = self.lrelu(self._inorm(x, 'in2'))
            x = self._conv(x, 128, k=3, s=2, name='c3')
            x = self.lrelu(self._inorm(x, 'in3'))
            x = self._conv(x, 256, k=3, s=2, name='c4')
            x = self.lrelu(self._inorm(x, 'in4'))
            gap = tf.reduce_mean(x, axis=[1, 2])
            out = tf.layers.dense(gap, 1, name='fc')
            return out

    # ------------------------------------------------------------------
    # Utility ops
    # ------------------------------------------------------------------
    @staticmethod
    def lrelu(x, leak=0.2):
        return tf.maximum(x, leak * x)

    @staticmethod
    def _conv(x, out_ch, k=3, s=1, name='conv'):
        return tf.layers.conv2d(
            x,
            out_ch,
            kernel_size=k,
            strides=s,
            padding='same',
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            bias_initializer=tf.constant_initializer(0.0),
            name=name,
        )

    @staticmethod
    def _inorm(x, scope):
        # Instance norm is more stable than batch norm for small batches
        return tf.contrib.layers.instance_norm(x, epsilon=1e-5, center=True, scale=True, scope=scope)

    @staticmethod
    def _resize_like(src, ref, method=tf.image.ResizeMethod.BICUBIC):
        ref_hw = tf.shape(ref)[1:3]
        return tf.image.resize_images(src, ref_hw, method=method)

    @staticmethod
    def high_pass_laplacian(img):
        # 3x3 Laplacian
        kernel = np.array([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]], dtype=np.float32)
        k = np.zeros((3, 3, 1, 1), dtype=np.float32)
        k[:, :, 0, 0] = kernel
        return tf.nn.conv2d(img, tf.convert_to_tensor(k), strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def _standardize_feature(x, eps=1e-6):
        # per-sample standardization over H,W
        mean, var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        return (x - mean) / tf.sqrt(var + eps)

    def _adv_weight_schedule(self, step):
        step_f = tf.cast(step, tf.float32)
        warm = float(self.adv_warmup_iters)
        ramp = float(max(1, self.adv_ramp_iters))

        def _zero():
            return tf.constant(0.0, tf.float32)

        def _ramp():
            t = (step_f - warm) / ramp
            t = tf.clip_by_value(t, 0.0, 1.0)
            return t * self.adv_weight_max

        return tf.cond(step_f < warm, _zero, _ramp)

    def _clip_grads(self, grads_and_vars, name='grad'):
        if self.grad_clip_norm is None or self.grad_clip_norm <= 0:
            return grads_and_vars

        grads = [g for g, v in grads_and_vars if g is not None]
        vars_ = [v for g, v in grads_and_vars if g is not None]
        if not grads:
            return grads_and_vars

        clipped, gn = tf.clip_by_global_norm(grads, self.grad_clip_norm)
        tf.summary.scalar(name + '_grad_norm', gn)
        return list(zip(clipped, vars_))
