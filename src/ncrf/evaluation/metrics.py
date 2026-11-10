"""NumPy-only validation metrics; no SciPy/OpenMP import in torch processes.

Contours use marching-square line segments. Distances integrate segment
midpoints with length weights (a documented discretization, not an exact
continuous Hausdorff distance). Coordinates are in the unit square.
"""
from __future__ import annotations

import numpy as np


def contour_segments(field):
    value = np.asarray(field, dtype=np.float64)
    height, width = value.shape
    segments = []
    # Vertex order clockwise: top-left, top-right, bottom-right, bottom-left.
    for y, x in np.argwhere((np.minimum.reduce([value[:-1, :-1], value[:-1, 1:], value[1:, 1:], value[1:, :-1]]) <= 0) &
                            (np.maximum.reduce([value[:-1, :-1], value[:-1, 1:], value[1:, 1:], value[1:, :-1]]) > 0)):
        v = [value[y, x], value[y, x+1], value[y+1, x+1], value[y+1, x]]
        p = [np.asarray([x, y]), np.asarray([x+1, y]), np.asarray([x+1, y+1]), np.asarray([x, y+1])]
        hits = {}
        for edge in range(4):
            j = (edge+1) % 4
            if (v[edge] > 0) != (v[j] > 0):
                alpha = v[edge] / (v[edge] - v[j])
                hits[edge] = p[edge] + alpha*(p[j]-p[edge])
        keys = list(hits)
        if len(keys) == 2:
            pairs = [(keys[0], keys[1])]
        elif len(keys) == 4:
            # Bilinear asymptotic decider, deterministic tie convention.
            q = v[0]*v[2] - v[1]*v[3]
            pairs = [(0, 1), (2, 3)] if q >= 0 else [(0, 3), (1, 2)]
        else:
            continue
        scale = np.asarray([width-1, height-1], dtype=np.float64)
        for a, b in pairs:
            if np.linalg.norm(hits[a]-hits[b]) > 1e-12:
                segments.append([hits[a]/scale, hits[b]/scale])
    return np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)


def directed_midpoint_distances(a, b):
    points = a.mean(axis=1)
    weights = np.sqrt(np.square(a[:, 1]-a[:, 0]).sum(axis=1))
    start, vector = b[:, 0], b[:, 1]-b[:, 0]
    denom = np.square(vector).sum(axis=1)
    result = []
    for offset in range(0, len(points), 128):
        relative = points[offset:offset+128, None, :] - start[None, :, :]
        t = np.clip((relative*vector).sum(axis=-1)/denom, 0, 1)
        distance = np.sqrt(np.square(relative-t[..., None]*vector).sum(axis=-1)).min(axis=1)
        result.extend(distance.tolist())
    return np.asarray(result), weights


def contour_metrics(prediction, truth):
    a, b = contour_segments(prediction), contour_segments(truth)
    missing = not len(a) or not len(b)
    if missing:
        # Both empty: zero; exactly one empty: unit-square diameter penalty.
        penalty = 0.0 if not len(a) and not len(b) else float(np.sqrt(2))
        return {"zero_contour_assd": penalty, "zero_contour_hd95": penalty,
                "prediction_contour_missing": int(not len(a)), "truth_contour_missing": int(not len(b))}
    ab, wa = directed_midpoint_distances(a, b)
    ba, wb = directed_midpoint_distances(b, a)
    distances, weights = np.concatenate((ab, ba)), np.concatenate((wa, wb))
    order = np.argsort(distances, kind="stable")
    weighted_cdf = np.cumsum(weights[order])/weights.sum()
    return {"zero_contour_assd": float((distances*weights).sum()/weights.sum()),
            "zero_contour_hd95": float(distances[order[np.searchsorted(weighted_cdf, 0.95)]]),
            "prediction_contour_missing": 0, "truth_contour_missing": 0}


def trapezoidal_node_weights(shape):
    """Return the VCH source-grid quadrature weights for a two-dimensional field."""
    height, width = shape
    if height < 2 or width < 2:
        raise ValueError("A source-grid field needs at least two nodes per direction.")
    weights = np.ones((height, width), dtype=np.float64)
    weights[[0, -1], :] *= 0.5
    weights[:, [0, -1]] *= 0.5
    return weights


def positive_phase_fraction(field, weights):
    """Trapezoid-weighted area fraction of the positive phase (phi > 0)."""
    value = np.asarray(field, dtype=np.float64)
    return float((weights * (value > 0.0)).sum() / weights.sum())


def neumann_spectral_length(field):
    """A characteristic length from the first spectral moment under even extension.

    VCH uses Neumann rather than periodic boundaries.  We therefore reflect the
    nodal field evenly at each boundary before the FFT, remove the zero mode,
    and define L=1/k1, where k1=sum(k*S(k))/sum(S(k)).  The unit-square grid
    spacing is retained, so this is a physical-domain length rather than a
    pixel count.  It is an evaluation diagnostic only; it is never optimized.
    """
    value = np.asarray(field, dtype=np.float64)
    height, width = value.shape
    if min(height, width) < 3:
        raise ValueError("Neumann spectral length needs at least three nodes per direction.")
    weights = trapezoidal_node_weights(value.shape)
    centered = value - (weights * value).sum() / weights.sum()
    reflected_y = np.concatenate((centered, centered[-2:0:-1, :]), axis=0)
    reflected = np.concatenate((reflected_y, reflected_y[:, -2:0:-1]), axis=1)
    power = np.abs(np.fft.fft2(reflected)) ** 2
    fy = np.fft.fftfreq(reflected.shape[0], d=1.0 / (height - 1))[:, None]
    fx = np.fft.fftfreq(reflected.shape[1], d=1.0 / (width - 1))[None, :]
    wave_number = np.sqrt(fx**2 + fy**2)
    nonzero = wave_number > 0.0
    denominator = float(power[nonzero].sum())
    if denominator <= np.finfo(np.float64).tiny:
        return float("inf")
    k1 = float((wave_number[nonzero] * power[nonzero]).sum() / denominator)
    return float(1.0 / k1)


def relative_neumann_spectral_length_error(prediction, truth):
    """Relative error between terminal Neumann-compatible spectral lengths."""
    predicted_length = neumann_spectral_length(prediction)
    truth_length = neumann_spectral_length(truth)
    if np.isinf(predicted_length) and np.isinf(truth_length):
        return 0.0
    if np.isinf(predicted_length) or np.isinf(truth_length):
        return 1.0
    return float(abs(predicted_length-truth_length) / max(abs(truth_length), np.finfo(np.float64).eps))


def field_metrics(prediction, truth, last_context):
    prediction, truth = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError("Expected [H,Y,X] prediction and truth on the source grid.")
    error = prediction-truth
    result = {"rmse": float(np.sqrt(np.square(error).mean())), "mae": float(np.abs(error).mean()),
              "terminal_rmse": float(np.sqrt(np.square(error[-1]).mean())),
              "terminal_phase_agreement": float(((prediction[-1] > 0) == (truth[-1] > 0)).mean())}
    for threshold in (0.1, 0.2, 0.3):
        a, b = np.abs(prediction[-1]) < threshold, np.abs(truth[-1]) < threshold
        denominator = a.sum()+b.sum()
        result[f"terminal_band_dice_{threshold:.1f}"] = float(2*(a & b).sum()/denominator) if denominator else 1.0
    weights = trapezoidal_node_weights(prediction.shape[-2:])
    mass = (prediction*weights).sum(axis=(-2, -1))/weights.sum()
    source_mass = float((np.asarray(last_context)*weights).sum()/weights.sum())
    result["source_grid_mass_abs_drift_mean"] = float(np.abs(mass-source_mass).mean())
    result["source_grid_mass_abs_drift_max"] = float(np.abs(mass-source_mass).max())
    n = prediction.shape[-1]-1
    def boundary(v):
        return n*np.concatenate((v[..., 1, :]-v[..., 0, :], v[..., -2, :]-v[..., -1, :],
                                 v[..., :, 1]-v[..., :, 0], v[..., :, -2]-v[..., :, -1]), axis=-1)
    result["boundary_normal_gradient_error_l1"] = float(np.abs(boundary(prediction)-boundary(truth)).mean())
    result.update(contour_metrics(prediction[-1], truth[-1]))
    result["terminal_positive_phase_fraction_abs_error"] = abs(
        positive_phase_fraction(prediction[-1], weights) - positive_phase_fraction(truth[-1], weights)
    )
    result["terminal_neumann_spectral_length_relative_error"] = relative_neumann_spectral_length_error(
        prediction[-1], truth[-1]
    )
    # Actual H-conditioned rollout per-frame errors, not separate short rollouts.
    for frame in (10, 20, 50, 90):
        result[f"rollout_frame{frame}_rmse"] = float(np.sqrt(np.square(error[frame-1]).mean())) if len(error) >= frame else None
    return result
