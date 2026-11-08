"""Image augmentations for pixel observations: random crop and color jitter."""

import functools

import jax
import jax.numpy as jnp

# Changed from upstream 7f48937: the unused augmentations (pmap variants, random flip,
# gaussian blur, solarize) are removed; random crop and color jitter stay.


def random_crop(key, img, padding):
    """Randomly crop one image after edge-padding it by `padding` pixels."""
    crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
    crop_from = jnp.concatenate([crop_from, jnp.zeros((2,), dtype=jnp.int32)])
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0), (0, 0)), mode="edge")
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


def batched_random_crop(key, imgs, padding=4):
    """Apply an independent random crop to every image in a batch."""
    keys = jax.random.split(key, imgs.shape[0])
    return jax.vmap(random_crop, (0, 0, None))(keys, imgs, padding)


def rgb_to_hsv(r, g, b):
    """Convert R, G, B values to H, S, V values.

    Reference TF implementation:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/adjust_saturation_op.cc
    Only input values between 0 and 1 are guaranteed to work properly, but this
    function complies with the TF implementation outside of this range.

    Args:
        r: A tensor representing the red color component as floats.
        g: A tensor representing the green color component as floats.
        b: A tensor representing the blue color component as floats.

    Returns:
        H, S, V values, each as tensors of shape [...] (same as the input without
        the last dimension).
    """
    vv = jnp.maximum(jnp.maximum(r, g), b)
    range_ = vv - jnp.minimum(jnp.minimum(r, g), b)
    sat = jnp.where(vv > 0, range_ / vv, 0.0)
    norm = jnp.where(range_ != 0, 1.0 / (6.0 * range_), 1e9)

    hr = norm * (g - b)
    hg = norm * (b - r) + 2.0 / 6.0
    hb = norm * (r - g) + 4.0 / 6.0

    hue = jnp.where(r == vv, hr, jnp.where(g == vv, hg, hb))
    hue = hue * (range_ > 0)
    hue = hue + (hue < 0)

    return hue, sat, vv


def hsv_to_rgb(h, s, v):
    """Convert H, S, V values to an R, G, B tuple.

    Reference TF implementation:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/adjust_saturation_op.cc
    Only input values between 0 and 1 are guaranteed to work properly, but this
    function complies with the TF implementation outside of this range.

    Args:
        h: A float tensor of arbitrary shape for the hue (0-1 values).
        s: A float tensor of the same shape for the saturation (0-1 values).
        v: A float tensor of the same shape for the value channel (0-1 values).

    Returns:
        An (r, g, b) tuple, each with the same dimension as the inputs.
    """
    c = s * v
    m = v - c
    dh = (h % 1.0) * 6.0
    fmodu = dh % 2.0
    x = c * (1 - jnp.abs(fmodu - 1))
    hcat = jnp.floor(dh).astype(jnp.int32)
    rr = jnp.where((hcat == 0) | (hcat == 5), c, jnp.where((hcat == 1) | (hcat == 4), x, 0)) + m
    gg = jnp.where((hcat == 1) | (hcat == 2), c, jnp.where((hcat == 0) | (hcat == 3), x, 0)) + m
    bb = jnp.where((hcat == 3) | (hcat == 4), c, jnp.where((hcat == 2) | (hcat == 5), x, 0)) + m
    return rr, gg, bb


def adjust_brightness(rgb_tuple, delta):
    """Add `delta` to every channel of an (r, g, b) tuple."""
    return jax.tree_util.tree_map(lambda x: x + delta, rgb_tuple)


def adjust_contrast(rgb_tuple, factor):
    """Scale each channel's deviation from its spatial mean by `factor`."""

    def _adjust_contrast_channel(channel):
        """Adjust the contrast of one channel."""
        mean = jnp.mean(channel, axis=(-2, -1), keepdims=True)
        return factor * (channel - mean) + mean

    return jax.tree_util.tree_map(_adjust_contrast_channel, rgb_tuple)


def adjust_saturation(h, s, v, factor):
    """Scale the saturation by `factor` and clip it to [0, 1]."""
    return h, jnp.clip(s * factor, 0.0, 1.0), v


def adjust_hue(h, s, v, delta):
    """Shift the hue by `delta`, wrapping around at 1."""
    # Note: this method exactly matches TF's adjust_hue (combined with the hsv/rgb
    # conversions) when running on GPU. When running on CPU, the results will be
    # different if all RGB values for a pixel are outside of the [0, 1] range.
    return (h + delta) % 1.0, s, v


def _random_brightness(rgb_tuple, rng, max_delta):
    """Apply a brightness shift drawn from [-max_delta, max_delta]."""
    delta = jax.random.uniform(rng, shape=(), minval=-max_delta, maxval=max_delta)
    return adjust_brightness(rgb_tuple, delta)


def _random_contrast(rgb_tuple, rng, max_delta):
    """Apply a contrast factor drawn from [1 - max_delta, 1 + max_delta]."""
    factor = jax.random.uniform(rng, shape=(), minval=1 - max_delta, maxval=1 + max_delta)
    return adjust_contrast(rgb_tuple, factor)


def _random_saturation(rgb_tuple, rng, max_delta):
    """Apply a saturation factor drawn from [1 - max_delta, 1 + max_delta]."""
    h, s, v = rgb_to_hsv(*rgb_tuple)
    factor = jax.random.uniform(rng, shape=(), minval=1 - max_delta, maxval=1 + max_delta)
    return hsv_to_rgb(*adjust_saturation(h, s, v, factor))


def _random_hue(rgb_tuple, rng, max_delta):
    """Apply a hue shift drawn from [-max_delta, max_delta]."""
    h, s, v = rgb_to_hsv(*rgb_tuple)
    delta = jax.random.uniform(rng, shape=(), minval=-max_delta, maxval=max_delta)
    return hsv_to_rgb(*adjust_hue(h, s, v, delta))


def _to_grayscale(image):
    """Convert an HWC RGB image to grayscale, repeated over 3 channels."""
    rgb_weights = jnp.array([0.2989, 0.5870, 0.1140])
    grayscale = jnp.tensordot(image, rgb_weights, axes=(-1, -1))[..., jnp.newaxis]
    return jnp.tile(grayscale, (1, 1, 3))  # Back to 3 channels.


def _color_transform_single_image(
    image,
    rng,
    brightness,
    contrast,
    saturation,
    hue,
    to_grayscale_prob,
    color_jitter_prob,
    apply_prob,
):
    """Apply color jittering and optional grayscaling to one HWC image."""
    apply_rng, transform_rng = jax.random.split(rng)
    perm_rng, b_rng, c_rng, s_rng, h_rng, cj_rng, gs_rng = jax.random.split(transform_rng, 7)

    # Whether the transform should be applied at all.
    should_apply = jax.random.uniform(apply_rng, shape=()) <= apply_prob
    # Whether to apply grayscale transform.
    should_apply_gs = jax.random.uniform(gs_rng, shape=()) <= to_grayscale_prob
    # Whether to apply color jittering.
    should_apply_color = jax.random.uniform(cj_rng, shape=()) <= color_jitter_prob

    def _make_cond(fn, idx):
        """Wrap `fn` so that it runs only when the jitter order selects `idx`."""

        def identity_fn(x, unused_rng, unused_param):
            """Return the input unchanged."""
            return x

        def cond_fn(args, i):
            """Apply `fn` with clipping if step `i` selects it, else the identity."""

            def clip(args):
                """Clip every array in `args` to [0, 1]."""
                return jax.tree_util.tree_map(lambda arg: jnp.clip(arg, 0.0, 1.0), args)

            out = jax.lax.cond(
                should_apply & should_apply_color & (i == idx),
                args,
                lambda a: clip(fn(*a)),
                args,
                lambda a: identity_fn(*a),
            )
            return jax.lax.stop_gradient(out)

        return cond_fn

    random_brightness_cond = _make_cond(_random_brightness, idx=0)
    random_contrast_cond = _make_cond(_random_contrast, idx=1)
    random_saturation_cond = _make_cond(_random_saturation, idx=2)
    random_hue_cond = _make_cond(_random_hue, idx=3)

    def _color_jitter(x):
        """Apply the enabled jitter transforms to one HWC image in order."""
        rgb_tuple = tuple(jax.tree_util.tree_map(jnp.squeeze, jnp.split(x, 3, axis=-1)))
        # Changed from upstream 7f48937: the shuffle option, always on, is removed; the four
        # transforms always run in a random order.
        order = jax.random.permutation(perm_rng, jnp.arange(4, dtype=jnp.int32))
        for idx in order:
            if brightness > 0:
                rgb_tuple = random_brightness_cond((rgb_tuple, b_rng, brightness), idx)
            if contrast > 0:
                rgb_tuple = random_contrast_cond((rgb_tuple, c_rng, contrast), idx)
            if saturation > 0:
                rgb_tuple = random_saturation_cond((rgb_tuple, s_rng, saturation), idx)
            if hue > 0:
                rgb_tuple = random_hue_cond((rgb_tuple, h_rng, hue), idx)
        return jnp.stack(rgb_tuple, axis=-1)

    out_apply = _color_jitter(image)
    out_apply = jax.lax.cond(
        should_apply & should_apply_gs, out_apply, _to_grayscale, out_apply, lambda x: x
    )
    return jnp.clip(out_apply, 0.0, 1.0)


def color_transform(
    rng,
    images,
    brightness=0.2,
    contrast=0.1,
    saturation=0.1,
    hue=0.03,
    color_jitter_prob=0.8,
    to_grayscale_prob=0.0,
    apply_prob=1.0,
):
    """Apply color jittering and/or grayscaling to a batch of images.

    Args:
        rng: a single PRNGKey.
        images: an (N, H, W, C, S) tensor with C=3; only index 0 of the last axis is used.
        brightness: the range of jitter on brightness.
        contrast: the range of jitter on contrast.
        saturation: the range of jitter on saturation.
        hue: the range of jitter on hue.
        color_jitter_prob: the probability of applying color jittering.
        to_grayscale_prob: the probability of converting the image to grayscale.
        apply_prob: the probability of applying the transform to a batch element.

    Returns:
        An (N, H, W, C, 1) tensor of the transformed images.
    """
    images = images[:, :, :, :, 0]
    rngs = jax.random.split(rng, images.shape[0])
    jitter_fn = functools.partial(
        _color_transform_single_image,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        color_jitter_prob=color_jitter_prob,
        to_grayscale_prob=to_grayscale_prob,
        apply_prob=apply_prob,
    )
    augmented_images = jax.vmap(jitter_fn)(images, rngs)
    return augmented_images[..., jnp.newaxis]
