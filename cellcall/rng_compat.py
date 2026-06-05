"""R-compatible random number generator using rpy2.

Calls R's sample() directly to produce identical permutation indices.
"""

from __future__ import annotations
import os
import numpy as np

# Set R_HOME before importing rpy2
if 'R_HOME' not in os.environ:
    # Try common R installation paths
    import glob
    for pattern in ['C:/Program Files/R/R-*/bin/Rscript.exe',
                    '/usr/bin/Rscript',
                    '/usr/local/bin/Rscript']:
        matches = glob.glob(pattern)
        if matches:
            os.environ['R_HOME'] = os.path.dirname(os.path.dirname(matches[0]))
            break

try:
    import rpy2.robjects as ro
    _HAS_RPY2 = True
except ImportError:
    _HAS_RPY2 = False


def generate_r_permutations(n_total: int, n_hit: int, n_perm: int, seed: int = 42) -> np.ndarray:
    """Generate permutation indices matching R's sample() exactly.

    Parameters
    ----------
    n_total : int
        Total number of genes (R: length(geneList)).
    n_hit : int
        Number of genes in gene set (R: length(gene.set)).
    n_perm : int
        Number of permutations.
    seed : int
        R's set.seed() value.

    Returns
    -------
    np.ndarray
        Shape (n_perm, n_hit), 0-based indices.
    """
    if not _HAS_RPY2:
        raise ImportError("rpy2 is required for R-compatible RNG. Install with: pip install rpy2")

    # Call R to generate all permutation indices at once
    r_code = f'''
    set.seed({seed})
    n_total <- {n_total}
    n_hit <- {n_hit}
    n_perm <- {n_perm}
    result <- matrix(0L, nrow=n_perm, ncol=n_hit)
    for (i in 1:n_perm) {{
        result[i,] <- sample.int(n_total, n_hit, replace=FALSE)
    }}
    result
    '''

    result = ro.r(r_code)
    # Convert to numpy (rpy2 returns 1-based indices)
    perm_indices = np.array(result, dtype=np.int64).reshape(n_perm, n_hit) - 1  # 0-based
    return perm_indices


class RCompatibleRNG:
    """R-compatible RNG for deterministic GSEA permutations."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._cache = {}  # Cache permutations by (n_total, n_hit)
        self._counter = {}  # Track how many permutations consumed per key

    def choice(self, n: int, size: int) -> np.ndarray:
        """Get next permutation (0-based indices), matching R's sample()."""
        key = (n, size)
        if key not in self._cache:
            # Generate a batch of 1000 permutations
            batch_size = max(1000, size * 2)
            self._cache[key] = generate_r_permutations(n, size, batch_size, self.seed)
            self._counter[key] = 0

        idx = self._counter[key]
        if idx >= len(self._cache[key]):
            # Generate more
            more = generate_r_permutations(n, size, 1000, self.seed + idx)
            self._cache[key] = np.vstack([self._cache[key], more])

        result = self._cache[key][idx]
        self._counter[key] = idx + 1
        return result
