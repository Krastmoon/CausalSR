
# -*- coding: utf-8 -*-
"""
CCM baseline benchmark for CP-FCD comparisons (single-scale method copied to 3 bands).

What this script does:
1) Generate multi-band synthetic data (Low/Mid/High) with ground-truth band-specific DAGs.
2) Run Convergent Cross Mapping (CCM) on the raw time series to estimate a single directed graph.
3) Copy this single graph to all 3 bands to compute structure metrics per band: F1, TPR, SHD.
4) Compute reconstruction metrics (standardized MSE/MAE) by:
   - CWT each variable once
   - For each band, replace coefficients using the estimated adjacency (same for all bands)
   - Concatenate the 3 bands back into a full W_pred and ICWT once per variable
5) Repeat for multiple random seeds, output per-run table + summary (mean±std).

Dependencies: numpy, pandas, pycwt, scipy (optional), igraph (optional; not required here), matplotlib (optional).
"""

import numpy as np
import pandas as pd
import pycwt as wavelet
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional


# -----------------------------
# Utils: standardization & metrics
# -----------------------------
def zscore_per_var(X: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X: [d,T] -> standardized Z, mean, std per variable."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    Z = (X - mu) / sd
    return Z, mu, sd


def mse_mae(X_true: np.ndarray, X_pred: np.ndarray) -> Tuple[float, float]:
    """Compute MSE/MAE over all entries (after standardization if desired)."""
    diff = X_true - X_pred
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    return mse, mae


def binarize_from_weights(W: np.ndarray, thresh: float = 1e-6) -> np.ndarray:
    B = (np.abs(W) > thresh).astype(int)
    np.fill_diagonal(B, 0)
    return B


def f1_tpr_shd(B_true: np.ndarray, B_est: np.ndarray) -> Dict[str, float]:
    """
    Directed edge metrics:
    - F1 for directed adjacency
    - TPR (recall) for directed edges
    - SHD computed on skeleton + reversals (common in causal discovery)
    """
    d = B_true.shape[0]
    # Directed TP/FP/FN
    tp = int(np.sum((B_true == 1) & (B_est == 1)))
    fp = int(np.sum((B_true == 0) & (B_est == 1)))
    fn = int(np.sum((B_true == 1) & (B_est == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)  # == TPR
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    # SHD: undirected extra + undirected missing + reversals
    # Skeletons (lower triangle)
    est_skel = np.tril(B_est + B_est.T, k=-1)  # 0/1/2
    true_skel = np.tril(B_true + B_true.T, k=-1)
    # Treat any nonzero as edge existence
    est_edges = (est_skel != 0).astype(int)
    true_edges = (true_skel != 0).astype(int)
    extra = int(np.sum((est_edges == 1) & (true_edges == 0)))
    missing = int(np.sum((est_edges == 0) & (true_edges == 1)))

    # reversals: edge exists in both skeletons but opposite direction
    rev = 0
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if B_est[i, j] == 1 and B_true[j, i] == 1 and B_true[i, j] == 0:
                rev += 1
    shd = extra + missing + rev

    return {"F1": float(f1), "TPR": float(recall), "SHD": float(shd)}


# -----------------------------
# Synthetic data generator (your original logic)
# -----------------------------
def sample_random_dag(d=3, p_edge=0.3, w_min=0.7, w_max=0.9, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    while True:
        order = rng.permutation(d)
        A = np.zeros((d, d))
        for u_rank in range(d):
            for v_rank in range(u_rank + 1, d):
                u, v = order[u_rank], order[v_rank]
                if rng.random() < p_edge:
                    A[u, v] = rng.uniform(w_min, w_max)
        if np.any(A != 0):
            np.fill_diagonal(A, 0.0)
            return A


def roll_shift(x, lag, dt=1.0, mode="zero"):
    n = len(x)
    if abs(lag - int(lag)) < 1e-9:
        lag = int(round(lag))
        y = np.roll(x, lag)
        if mode == "zero" and lag > 0:
            y[:lag] = 0.0
        return y
    tau = lag * dt
    Xf = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, dt)
    phase_shift = np.exp(-2j * np.pi * freqs * tau)
    y = np.fft.irfft(Xf * phase_shift, n=n)
    return y


def _smooth_env(T, rng, smooth_len=256):
    e = rng.normal(0, 1, size=T)
    if smooth_len > 1:
        k = np.hanning(smooth_len)
        k /= k.sum()
        e = np.convolve(e, k, mode="same")
    e -= e.min()
    if e.max() > 0:
        e /= e.max()
    return 0.5 + e


def _band_limited_noise(center_hz, bw_hz, fs, T, rng):
    n = T
    freqs = np.fft.rfftfreq(n, 1 / fs)
    sigma = bw_hz / 2.355 if bw_hz > 0 else (center_hz / 10 + 1e-6)
    win = np.exp(-0.5 * ((freqs - center_hz) / (sigma + 1e-12)) ** 2)
    phase = rng.uniform(0, 2 * np.pi, size=len(freqs))
    amp = rng.normal(0, 1, size=len(freqs))
    X = win * amp * (np.cos(phase) + 1j * np.sin(phase))
    x = np.fft.irfft(X, n=n)
    if np.std(x) > 1e-12:
        x /= np.std(x)
    return x


def synth_base_complex(d, T, fs, freqs, phi_shared=0.0, rng=None,
                       mode="narrowband_noise",
                       amp_range=(0.9, 1.1),
                       max_harm=3, harmonics_decay=0.6,
                       am_strength=0.4, fm_strength=0.03, env_smooth_len=256,
                       bandwidth_ratio=0.15,
                       chirp_ratio=0.1,
                       target_abs_corr=0.25, max_decor_iter=3):
    rng = np.random.default_rng() if rng is None else rng
    t = np.arange(T) / fs
    Z = np.zeros((d, T), float)

    for i in range(d):
        f0 = float(freqs[i])
        a = rng.uniform(*amp_range)

        if mode == "harmonics":
            z = np.zeros(T)
            for k in range(1, max_harm + 1):
                ak = a * (harmonics_decay ** (k - 1))
                dphi = rng.normal(0, 0.03)
                z += ak * np.sin(2 * np.pi * (k * f0) * t + phi_shared + dphi)
            z += rng.normal(0, 0.03, size=T)

        elif mode == "amfm":
            env = _smooth_env(T, rng, env_smooth_len)
            m = _smooth_env(T, rng, env_smooth_len) - 1.0
            inst_f = f0 * (1.0 + fm_strength * m)
            phase = 2 * np.pi * np.cumsum(inst_f) / fs + phi_shared
            z = a * (1.0 + am_strength * (env - 1.0)) * np.sin(phase)
            z += rng.normal(0, 0.02, size=T)

        elif mode == "narrowband_noise":
            bw = max(f0 * bandwidth_ratio, fs / T)
            nb = _band_limited_noise(f0, bw, fs, T, rng)
            carrier = np.sin(2 * np.pi * f0 * t + phi_shared)
            z = a * (0.7 * nb + 0.3 * carrier)
            z += rng.normal(0, 0.02, size=T)

        elif mode == "chirp":
            f1, f2 = f0 * (1 - chirp_ratio), f0 * (1 + chirp_ratio)
            inst_f = np.linspace(f1, f2, T)
            phase = 2 * np.pi * np.cumsum(inst_f) / fs + phi_shared
            z = a * np.sin(phase)
            z += rng.normal(0, 0.02, size=T)

        else:
            raise ValueError(f"unknown base mode: {mode}")

        std = np.std(z)
        if std > 1e-12:
            z /= std
        Z[i] = z

    for _ in range(max_decor_iter):
        C = np.corrcoef(Z)
        hi = np.abs(C - np.eye(d)).max()
        if hi <= target_abs_corr:
            break
        Z += rng.normal(0, 0.02, size=Z.shape)

    return Z


def generate_data_simple(duration=20, fs=500, d=3,
                         band_ranges=None, noise_std=0.1, rng=None,
                         base_mode="narrowband_noise",
                         phi_shared_map=None):
    if band_ranges is None:
        band_ranges = {"Low": (0.1, 6), "Mid": (7, 15), "High": (17, 25)}
    rng = np.random.default_rng() if rng is None else rng
    T = int(duration * fs)
    dt = 1.0 / fs
    t = np.arange(T) / fs

    freqs_dict = {"Low": [2, 3, 4], "Mid": [11, 12, 13], "High": [21, 22, 23]}
    if phi_shared_map is None:
        phi_shared_map = {k: 0.0 for k in band_ranges}

    A_bands, tau_bands, comps = {}, {}, {}
    X_total = np.zeros((d, T))

    for band in band_ranges:
        A = sample_random_dag(d, rng=rng)
        A_bands[band] = A

        f = np.array(freqs_dict[band], dtype=float)
        Z = synth_base_complex(d, T, fs, f, phi_shared=phi_shared_map[band],
                               rng=rng, mode=base_mode)

        tau = np.zeros((d, d), dtype=float)
        for i in range(d):
            max_tau = fs / f[i] / 8.0
            for j in range(d):
                if i == j or A[i, j] == 0:
                    tau[i, j] = 0.0
                else:
                    tau[i, j] = rng.uniform(1.0, max(1.0, max_tau))
        tau_bands[band] = tau.copy()

        X_band = Z.copy()
        for i in range(d):      # source
            for j in range(d):  # target
                if i != j and A[i, j] != 0:
                    lag = tau[i, j]
                    X_band[j] += A[i, j] * roll_shift(Z[i], lag, dt=dt, mode="zero")

        comps[band] = X_band
        X_total += X_band

    X_total += rng.normal(0, noise_std, size=X_total.shape)
    data = {f"x{i + 1}": X_total[i] for i in range(d)}
    return data, t, X_total, A_bands, tau_bands, comps, dt


# -----------------------------
# CCM implementation (lightweight)
# -----------------------------
def _embed(ts: np.ndarray, E: int, tau: int) -> Tuple[np.ndarray, np.ndarray]:
    """Takens embedding. Return M [N, E] and time index t_idx matching original series."""
    T = len(ts)
    start = (E - 1) * tau
    N = T - start
    M = np.zeros((N, E), dtype=float)
    for e in range(E):
        M[:, e] = ts[start - e * tau: T - e * tau]
    t_idx = np.arange(start, T)
    return M, t_idx


def _ccm_skill(source_ts: np.ndarray, target_ts: np.ndarray,
               E: int = 3, tau: int = 1,
               lib_size: Optional[int] = None,
               k: Optional[int] = None,
               seed: int = 0) -> float:
    """
    Cross map skill rho(target | M_source).
    - Build manifold from source
    - Predict target via locally weighted neighbors in source manifold
    - Return Pearson correlation between predicted and true target
    """
    rng = np.random.default_rng(seed)
    M, t_idx = _embed(source_ts, E, tau)
    y = target_ts[t_idx]

    N = M.shape[0]
    if lib_size is None or lib_size > N:
        lib_size = N
    if k is None:
        k = E + 1
    k = min(k, lib_size - 1)

    # Choose a library subset (random without replacement)
    lib_idx = rng.choice(N, size=lib_size, replace=False)

    # Precompute distances within library
    M_lib = M[lib_idx]
    y_lib = y[lib_idx]
    y_hat = np.zeros(lib_size, dtype=float)

    # Brute-force KNN (OK for small d, small T)
    for ii, q in enumerate(range(lib_size)):
        v = M_lib[q]
        # distances to all other points in library
        dists = np.linalg.norm(M_lib - v[None, :], axis=1)
        dists[q] = np.inf  # exclude itself
        nn = np.argsort(dists)[:k]
        d1 = dists[nn[0]]
        if not np.isfinite(d1) or d1 <= 1e-12:
            w = np.ones_like(nn, dtype=float) / len(nn)
        else:
            w = np.exp(-dists[nn] / d1)
            w = w / (np.sum(w) + 1e-12)
        y_hat[ii] = np.sum(w * y_lib[nn])

    y_true = y_lib
    # Pearson correlation
    yt = y_true - y_true.mean()
    yh = y_hat - y_hat.mean()
    denom = (np.sqrt(np.sum(yt ** 2)) * np.sqrt(np.sum(yh ** 2)) + 1e-12)
    rho = float(np.sum(yt * yh) / denom)
    return rho


def ccm_direction_matrix(X: np.ndarray,
                         E: int = 3, tau: int = 1,
                         lib_frac: float = 0.8,
                         rho_threshold: float = 0.1,
                         margin: float = 0.02,
                         seed: int = 0) -> np.ndarray:
    """
    Estimate a directed adjacency matrix using CCM on raw time series.
    Rule:
      - compute rho(j | i) and rho(i | j)
      - if rho(j|i) >= rho_threshold and rho(j|i) > rho(i|j) + margin => i -> j
      - else no directed edge
    """
    d, T = X.shape
    N = T - (E - 1) * tau
    lib_size = max(int(lib_frac * N), E + 2)

    W = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            rho_j_from_i = _ccm_skill(X[i], X[j], E=E, tau=tau, lib_size=lib_size, seed=seed + 13 * i + 7 * j)
            rho_i_from_j = _ccm_skill(X[j], X[i], E=E, tau=tau, lib_size=lib_size, seed=seed + 13 * j + 7 * i)
            if rho_j_from_i >= rho_threshold and (rho_j_from_i > rho_i_from_j + margin):
                W[i, j] = max(rho_j_from_i, 0.0)
    return W


# -----------------------------
# Reconstruction via CWT band replacement + single ICWT (your correct logic)
# -----------------------------
def compute_cwt_all(data: Dict[str, np.ndarray],
                    variables: List[str],
                    dt: float,
                    mother=None,
                    dj: float = 1/12,
                    s0: Optional[float] = None,
                    J: Optional[int] = None) -> Dict[str, Dict[str, np.ndarray]]:
    if mother is None:
        mother = wavelet.Morlet(6)
    if s0 is None:
        s0 = 2 * dt
    if J is None:
        J = int(7 / dj)

    out = {}
    for var in variables:
        W, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(data[var], dt, dj=dj, s0=s0, J=J, wavelet=mother)
        out[var] = {"W": W, "scales": scales, "freqs": freqs, "coi": coi}
    return out


def reconstruct_from_bands(A_bands: Dict[str, np.ndarray],
                           cwt_results: Dict[str, Dict[str, np.ndarray]],
                           variables: List[str],
                           freq_bands: Dict[str, Tuple[float, float]],
                           dt: float, dj: float, mother) -> np.ndarray:
    """
    For each variable j:
      W_pred_j = W_orig_j
      For each band k: replace W_pred_j[band] by sum_i A_k[i,j] * W_i[band]
      Then icwt once => x_pred_j (real)
    """
    d = len(variables)
    freqs = cwt_results[variables[0]]["freqs"]
    scales = cwt_results[variables[0]]["scales"]
    T = cwt_results[variables[0]]["W"].shape[1]
    X_pred = np.zeros((d, T), dtype=float)

    for j, var_j in enumerate(variables):
        W_pred_j = np.array(cwt_results[var_j]["W"], copy=True)  # complex
        for band_name, (fmin, fmax) in freq_bands.items():
            fmask = (freqs >= fmin) & (freqs <= fmax)
            if not np.any(fmask):
                continue
            A_k = A_bands[band_name]  # [d,d]
            W_band = np.zeros_like(W_pred_j[fmask, :], dtype=np.complex128)
            for i, var_i in enumerate(variables):
                W_i = cwt_results[var_i]["W"]
                W_band += A_k[i, j] * W_i[fmask, :]
            W_pred_j[fmask, :] = W_band

        x_pred_j = wavelet.icwt(W_pred_j, scales, dt, dj, mother)
        # icwt should be (almost) real; keep real part for robustness
        X_pred[j] = np.real(x_pred_j)
    return X_pred


# -----------------------------
# Benchmark runner
# -----------------------------
@dataclass
class BenchmarkConfig:
    n_runs: int = 10
    d: int = 3
    duration: float = 20.0
    fs: int = 500
    noise_std: float = 0.1
    base_mode: str = "narrowband_noise"
    band_ranges: Dict[str, Tuple[float, float]] = None

    # CCM params
    E: int = 3
    tau: int = 1
    lib_frac: float = 0.8
    rho_threshold: float = 0.1
    margin: float = 0.02

    # Reconstruction (CWT)
    dj: float = 1/12
    morlet_w0: float = 6.0


def run_ccm_benchmark(cfg: BenchmarkConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if cfg.band_ranges is None:
        cfg.band_ranges = {"Low": (0.1, 6), "Mid": (7, 15), "High": (17, 25)}
    variables = [f"x{i+1}" for i in range(cfg.d)]
    mother = wavelet.Morlet(cfg.morlet_w0)

    rows = []
    for run in range(cfg.n_runs):
        rng = np.random.default_rng(1000 + run)

        data, t, X_total, A_true_bands, _, _, dt = generate_data_simple(
            duration=cfg.duration, fs=cfg.fs, d=cfg.d,
            band_ranges=cfg.band_ranges, noise_std=cfg.noise_std,
            rng=rng, base_mode=cfg.base_mode
        )

        # CCM on raw series
        W_est = ccm_direction_matrix(
            X_total, E=cfg.E, tau=cfg.tau, lib_frac=cfg.lib_frac,
            rho_threshold=cfg.rho_threshold, margin=cfg.margin, seed=2000 + run
        )
        B_est = binarize_from_weights(W_est)

        # Copy to 3 bands for structure metrics
        for band_name in cfg.band_ranges.keys():
            B_true = binarize_from_weights(A_true_bands[band_name])
            m = f1_tpr_shd(B_true, B_est)
            rows.append({
                "run": run,
                "band": band_name,
                "F1": m["F1"],
                "TPR": m["TPR"],
                "SHD": m["SHD"]
            })

        # Reconstruction metrics (standardized)
        # Standardize true series first
        Xz, mu, sd = zscore_per_var(X_total)

        # CWT computed on standardized signals
        data_z = {variables[i]: Xz[i] for i in range(cfg.d)}
        cwt_res = compute_cwt_all(data_z, variables, dt=dt, mother=mother, dj=cfg.dj)

        # Use same adjacency for all bands
        A_bands_est = {bn: W_est.copy() for bn in cfg.band_ranges.keys()}
        X_pred = reconstruct_from_bands(A_bands_est, cwt_res, variables, cfg.band_ranges, dt=dt, dj=cfg.dj, mother=mother)
        # Pred is in standardized scale already
        mse, mae = mse_mae(Xz, X_pred)

        # store one row for reconstruction (band = "ALL")
        rows.append({
            "run": run,
            "band": "ALL",
            "F1": np.nan,
            "TPR": np.nan,
            "SHD": np.nan,
            "MSE": mse,
            "MAE": mae
        })

    df = pd.DataFrame(rows)

    # Summary: structure metrics by band; recon metrics overall
    summary_parts = []
    for band_name in list(cfg.band_ranges.keys()) + ["ALL"]:
        sub = df[df["band"] == band_name]
        stats = {"band": band_name}
        for col in ["F1", "TPR", "SHD", "MSE", "MAE"]:
            if col in sub.columns and sub[col].notna().any():
                stats[col + "_mean"] = float(sub[col].mean())
                stats[col + "_std"] = float(sub[col].std(ddof=1)) if len(sub[col].dropna()) > 1 else 0.0
        summary_parts.append(stats)

    summary = pd.DataFrame(summary_parts)
    return df, summary


def main():
    cfg = BenchmarkConfig(n_runs=30, d=3)
    df, summary = run_ccm_benchmark(cfg)
    print("\nPer-run metrics (first 20 rows):")
    print(df.head(20))
    print("\nSummary (mean±std):")
    with pd.option_context("display.max_columns", 200):
        print(summary)


if __name__ == "__main__":
    main()
