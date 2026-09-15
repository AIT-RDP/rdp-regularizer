import numpy as np

from .util import nan_bfill, nan_ffill


def soft_threshold_svd_reconstruct(
        values: np.ndarray, max_rank: int | None = None,
        shrinkage: float | None = None,
        max_iter: int = 200, tol: float = 1e-5
    ) -> np.ndarray:
    """
    Complete a matrix with missing entries (NaN) by iterative
    soft-thresholded SVD.

    Each step: SVD the current estimate, shrink singular values by
    `shrinkage` (and truncate to `max_rank`), then reset the observed
    entries to their true values. Repeat to convergence.
    """

    values = np.asarray(values, dtype=float)
    missing = np.isnan(values) # boolean mask of the entries to estimate
    if not missing.any(): # nothing missing -> nothing to do
        return values.copy()

    # Initial guess: SVD needs a complete matrix, so seed the missing cells before iterating.
    # We use each column's mean (a sensible starting profile); if a whole column is empty,
    # np.nanmean is NaN and we fall back to the global mean.
    filled = values.copy()
    col_mean = np.nanmean(values, axis=0) # per-column mean, ignoring NaN
    col_mean = np.where(np.isnan(col_mean), np.nanmean(values), col_mean) # empty-column guard

    # place each missing cell's column mean into it: np.where(missing)[1] is the column index
    # of every missing entry, in the same order as filled[missing].
    filled[missing] = np.take(col_mean, np.where(missing)[1])

    # Auto-pick the shrinkage strength if not supplied. Rule of thumb from the reference
    # implementation: a small fraction of the largest singular value (larger -> more smoothing).
    if shrinkage is None:
        s0 = np.linalg.svd(filled, compute_uv=False)      # singular values only
        shrinkage = s0[0] / 50.0                           # s0[0] is the largest singular value

    # Iterate: SVD -> shrink -> reset observed -> repeat
    prev = filled.copy() # previous estimate, for the convergence check
    for _ in range(max_iter):
        # 1) factor the current complete estimate into patterns (U, Vt) and
        #    their strengths (singular values s), strongest first.
        U, s, Vt = np.linalg.svd(filled, full_matrices=False)

        # 2) soft-threshold: subtract `shrinkage` from every singular value and
        #    floor at 0. Patterns weaker than `shrinkage` vanish -> low rank.
        s_shrunk = np.maximum(s - shrinkage, 0.0)
        if max_rank is not None:
            s_shrunk[max_rank:] = 0.0      # hard cap: keep at most `max_rank` patterns

        # 3) rebuild a low-rank approximation from the surviving patterns.
        #    (U * s_shrunk) scales each column of U by its singular value.
        recon = (U * s_shrunk) @ Vt

        # 4) keep the real data exactly; only overwrite the missing cells with
        #    the low-rank estimate. This is what pulls the fit toward the data.
        filled = values.copy()               # observed entries = truth
        filled[missing] = recon[missing]  # missing entries = low-rank estimate

        # 5) stop once the estimate barely changes (relative Frobenius norm).
        denom = np.linalg.norm(prev) + 1e-12   # +eps guards against divide-by-zero
        if np.linalg.norm(filled - prev) / denom < tol:
            break

        prev = filled.copy() # update the previous estimate

    return filled


def knn_reconstruct(
        values: np.ndarray, k: int = 5
    ) -> np.ndarray:
    """
    Fill missing entries (NaN) from the k nearest ROWS.

    Distance between two rows uses only columns observed in BOTH
    (so gaps don't break it). Each missing cell is a distance-weighted
    average of that column's value over the k nearest rows that have it.
    Rows with no overlap fall back to the column mean.
    """
    values = np.asarray(values, dtype=float)
    missing = np.isnan(values) # boolean mask of entries to estimate
    if not missing.any(): # nothing missing -> nothing to do
        return values.copy()

    n, _ = values.shape # n = number of rows (days)

    # Column means, used only as a last-resort fallback for a row that shares
    # no observed columns with any other row (e.g. a fully-missing day).
    col_mean = np.nanmean(values, axis=0)
    col_mean = np.where(np.isnan(col_mean), np.nanmean(values), col_mean) # empty-column guard
    out = values.copy() # result; observed entries stay untouched

    # Process one row (day) at a time.
    for i in range(n):
        cols = np.where(missing[i])[0] # which columns of this row need filling
        if cols.size == 0: # this row is complete -> skip
            continue

        # Distance from row i to every other row: Compared only over columns BOTH rows observe,
        # so gaps don't break it. RMS (root-mean-square) normalizes by the count of shared columns,
        # so rows sharing few columns aren't unfairly judged closer or farther.
        dist = np.full(n, np.inf) # inf = "not comparable / itself"
        for j in range(n):
            if j == i: # a row is not its own neighbor
                continue
            shared = ~missing[i] & ~missing[j] # columns observed in both rows
            if shared.any(): # need at least one shared column
                dist[j] = np.sqrt(np.mean((values[i, shared] - values[j, shared]) ** 2))

        order = np.argsort(dist) # row indices sorted nearest -> farthest

        # Fill each missing column of this row.
        for c in cols:
            # nearest rows that (a) are comparable and (b) actually have column c
            nn = [j for j in order if np.isfinite(dist[j]) and not missing[j, c]][:k]
            if not nn: # no usable neighbor -> fall back to mean
                out[i, c] = col_mean[c]
                continue
            # weighted average: closer rows count more. +1e-8 avoids div-by-zero
            # when a neighbor is at distance 0 (identical shared values).
            w = 1.0 / (dist[nn] + 1e-8) # inverse-distance weights
            out[i, c] = np.sum(w * values[nn, c]) / np.sum(w) # normalized weighted mean

    return out


def daily_naive_reconstruct(
        X: np.ndarray
    ) -> np.ndarray:
    """
    Fill missing entries (NaN) from the daily naive approach.
    """
    return nan_bfill(nan_ffill(X, axis=0), axis=0)


def linear_interpolation(values: np.ndarray) -> np.ndarray:
    """
    Fill missing entries (NaN) from the linear interpolation.
    """
    x = np.arange(len(values))
    y = np.asarray(values, dtype=np.float64).copy()

    # Sanity check: if there are less than 2 valid values, return the original array.
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return y

    # Find the first and last valid values.
    first, last = np.flatnonzero(valid)[0], np.flatnonzero(valid)[-1]
    # Create a mask for the interior values.
    interior = ~valid
    # Set the first and last valid values to False so they are not interpolated.
    interior[:first] = False
    interior[last + 1:] = False

    # Use numpy's interpolation function to fill the interior values.
    y[interior] = np.interp(x[interior], x[valid], y[valid])

    return y
