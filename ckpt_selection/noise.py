"""Shape pi0's noise input from the DSRL actor's output (App. A "Policy and learner").

The actor emits one 32-dimensional latent per decision. pi0 takes a 50 x 32 noise input. DSRL pads
the latent chunk with copies of its last row, as upstream train_utils_sim.py does. With one latent
per decision, this repeats the latent across every row. The GPU modules use this helper for scan
and confirm rollouts, held-out statistics and the SoftOPC pool.
"""

import numpy as np

PI0_NOISE_ROWS = 50  # pi0 action horizon (App. A "50x32 noise input"); agent.py checks the model
PI0_NOISE_DIM = 32  # App. A "Its 32-dimensional noise vector"


def fill_noise_rows(latent_chunk: np.ndarray) -> np.ndarray:
    """Pad a (k, dim) latent chunk to PI0_NOISE_ROWS rows with its last row; add a batch axis.

    Returns an array of shape (1, PI0_NOISE_ROWS, dim), the layout of pi0's noise argument.
    """
    latent_chunk = np.asarray(latent_chunk)
    if latent_chunk.ndim != 2 or latent_chunk.shape[0] > PI0_NOISE_ROWS:
        expected = f"(k <= {PI0_NOISE_ROWS}, dim)"
        raise ValueError(f"expected a {expected} latent chunk, got {latent_chunk.shape}")
    padding = np.repeat(latent_chunk[-1:], PI0_NOISE_ROWS - latent_chunk.shape[0], axis=0)
    return np.concatenate([latent_chunk, padding], axis=0)[None]
