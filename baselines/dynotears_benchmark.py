import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import pycwt as wavelet
from scipy.linalg import expm
from scipy.optimize import minimize


# ==========================================================
# Utils: seed, metrics, binarization, NOTEARS acyclicity
# ==========================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    sd = x.std()
    return (x - mu) / (sd + eps)


def safe_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if np.any(~np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def binarize(W: np.ndarray, thr: float = 0.3) -> np.ndarray:
    B = (np.abs(W) > thr).astype(int)
    np.fill_diagonal(B, 0)
    return B


def f1_from_adj(B_true: np.ndarray, B_est: np.ndarray) -> float:
    B_true = (B_true != 0).astype(int)
    B_est = (B_est != 0).astype(int)
    tp = np.sum((B_true == 1) & (B_est == 1))
    fp = np.sum((B_true == 0) & (B_est == 1))
    fn = np.sum((B_true == 1) & (B_est == 0))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def shd_tpr_from_adj(B_true: np.ndarray, B_est: np.ndarray) -> Tuple[int, float]:
    """SHD on directed adjacency (counts extra+missing+reversed)."""
    B_true = (B_true != 0).astype(int)
    B_est = (B_est != 0).astype(int)
    d = B_true.shape[0]

    # True positives for directed edges
    tp = np.sum((B_true == 1) & (B_est == 1))
    cond = np.sum(B_true)
    tpr = tp / max(cond, 1)

    # SHD computation
    # skeleton indices
    pred_skel = (B_est + B_est.T) > 0
    true_skel = (B_true + B_true.T) > 0

    extra = np.sum(pred_skel & (~true_skel))
    missing = np.sum((~pred_skel) & true_skel)

    # reversals: edge exists in both skeletons but wrong direction
    rev = 0
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if B_true[i, j] == 1 and B_est[i, j] == 0 and B_est[j, i] == 1:
                rev += 1

    # extra/missing were counted on full matrix; convert to undirected counts
    extra //= 2
    missing //= 2

    shd = int(extra + missing + rev)
    return shd, float(tpr)


def mse_mae_standardized(x_true: np.ndarray, x_pred: np.ndarray) -> Tuple[float, float]:
    """Compute MSE/MAE after z-score standardization per variable."""
    x_true = np.asarray(x_true, dtype=float)
    x_pred = np.asarray(x_pred, dtype=float)
    d = x_true.shape[0]
    mses, maes = [], []
    for i in range(d):
        yt = zscore(x_true[i])
        yp = zscore(x_pred[i])
        diff = yp - yt
        mses.append(float(np.mean(diff ** 2)))
        maes.append(float(np.mean(np.abs(diff))))
    return float(np.mean(mses)), float(np.mean(maes))


def h_func(A: np.ndarray) -> float:
    """NOTEARS differentiable acyclicity constraint: tr(expm(A∘A)) - d."""
    d = A.shape[0]
    return float(np.trace(expm(A * A)) - d)


# ==========================================================
# Synthetic multi-band generator (same spirit as CP-FCD)
# ==========================================================

def sample_random_dag(d: int = 3, p_edge: float = 0.3,
                      w_min: float = 0.7, w_max: float = 0.9,
                      rng: np.random.Generator | None = None) -> np.ndarray:
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
            return A


def roll_shift(x: np.ndarray, lag: float, dt: float = 1.0, mode: str = "zero") -> np.ndarray:
    n = len(x)
    if abs(lag - int(lag)) < 1e-9:
        lag_i = int(round(lag))
        y = np.roll(x, lag_i)
        if mode == "zero" and lag_i > 0:
            y[:lag_i] = 0.0
        return y

    tau = lag * dt
    Xf = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, dt)
    phase_shift = np.exp(-2j * np.pi * freqs * tau)
    Yf = Xf * phase_shift
    y = np.fft.irfft(Yf, n=n)
    return y


def _smooth_env(T: int, rng: np.random.Generator, smooth_len: int = 256) -> np.ndarray:
    e = rng.normal(0, 1, size=T)
    if smooth_len > 1:
        k = np.hanning(smooth_len)
        k /= k.sum()
        e = np.convolve(e, k, mode="same")
    e -= e.min()
    if e.max() > 0:
        e /= e.max()
    return 0.5 + e


def _band_limited_noise(center_hz: float, bw_hz: float, fs: float, T: int, rng: np.random.Generator) -> np.ndarray:
    freqs = np.fft.rfftfreq(T, 1 / fs)
    sigma = bw_hz / 2.355 if bw_hz > 0 else (center_hz / 10 + 1e-6)
    win = np.exp(-0.5 * ((freqs - center_hz) / (sigma + 1e-12)) ** 2)
    phase = rng.uniform(0, 2 * np.pi, size=len(freqs))
    amp = rng.normal(0, 1, size=len(freqs))
    X = win * amp * (np.cos(phase) + 1j * np.sin(phase))
    x = np.fft.irfft(X, n=T)
    if np.std(x) > 1e-12:
        x = x / np.std(x)
    return x


def synth_base_complex(d: int, T: int, fs: float, freqs: np.ndarray,
                       phi_shared: float = 0.0,
                       rng: np.random.Generator | None = None,
                       mode: str = "narrowband_noise",
                       bandwidth_ratio: float = 0.15) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    t = np.arange(T) / fs
    Z = np.zeros((d, T), float)

    for i in range(d):
        f0 = float(freqs[i])
        if mode == "narrowband_noise":
            bw = max(f0 * bandwidth_ratio, fs / T)
            nb = _band_limited_noise(f0, bw, fs, T, rng)
            carrier = np.sin(2 * np.pi * f0 * t + phi_shared)
            z = 0.7 * nb + 0.3 * carrier + rng.normal(0, 0.02, size=T)
        else:
            z = np.sin(2 * np.pi * f0 * t + phi_shared) + rng.normal(0, 0.02, size=T)

        if np.std(z) > 1e-12:
            z = z / np.std(z)
        Z[i] = z

    return Z


def generate_data_simple(duration: float = 20, fs: int = 500, d: int = 3,
                         band_ranges: Dict[str, Tuple[float, float]] | None = None,
                         noise_std: float = 0.1,
                         rng: np.random.Generator | None = None,
                         base_mode: str = "narrowband_noise",
                         phi_shared_map: Dict[str, float] | None = None) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    rng = np.random.default_rng() if rng is None else rng
    if band_ranges is None:
        band_ranges = {"Low": (0.1, 6), "Mid": (7, 15), "High": (17, 25)}

    T = int(duration * fs)
    dt = 1.0 / fs
    t = np.arange(T) / fs

    freqs_dict = {"Low": [2, 3, 4], "Mid": [11, 12, 13], "High": [21, 22, 23]}
    if phi_shared_map is None:
        phi_shared_map = {k: 0.0 for k in band_ranges}

    A_bands: Dict[str, np.ndarray] = {}
    X_total = np.zeros((d, T))

    for band, _ in band_ranges.items():
        A = sample_random_dag(d, rng=rng)
        A_bands[band] = A

        f = np.array(freqs_dict[band], dtype=float)
        Z = synth_base_complex(d, T, fs, f, phi_shared=phi_shared_map[band], rng=rng, mode=base_mode)

        tau = np.zeros((d, d), dtype=float)
        for i in range(d):
            max_tau = fs / f[i] / 8.0
            for j in range(d):
                if i == j or A[i, j] == 0:
                    continue
                tau[i, j] = rng.uniform(1.0, max(1.0, max_tau))

        X_band = Z.copy()
        for i in range(d):
            for j in range(d):
                if i != j and A[i, j] != 0:
                    X_band[j] += A[i, j] * roll_shift(Z[i], tau[i, j], dt=dt, mode="zero")

        X_total += X_band

    X_total += rng.normal(0, noise_std, size=X_total.shape)
    data = {f"x{i + 1}": X_total[i] for i in range(d)}
    return data, t, X_total, A_bands


# ==========================================================
# Wavelet transform helpers + reconstruction
# ==========================================================

def compute_cwt(data: Dict[str, np.ndarray], variables: List[str], dt: float,
                dj: float, s0: float, J: int, mother) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for var in variables:
        W, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(data[var], dt, dj, s0, J, mother)
        out[var] = {"W": W, "scales": scales, "freqs": freqs, "coi": coi}
    return out


def reconstruct_from_bands(A_bands_est: Dict[str, np.ndarray],
                           cwt_results: Dict[str, Dict[str, np.ndarray]],
                           variables: List[str],
                           freq_bands: Dict[str, Tuple[float, float]],
                           dt: float, dj: float, mother) -> np.ndarray:
    """Reconstruct full time series per variable by band-wise coefficient replacement then ONE icwt."""
    d = len(variables)
    freqs = cwt_results[variables[0]]["freqs"]
    scales = cwt_results[variables[0]]["scales"]
    T = cwt_results[variables[0]]["W"].shape[1]
    X_pred = np.zeros((d, T), dtype=float)

    for j, var_j in enumerate(variables):
        W_pred_j = cwt_results[var_j]["W"].copy()

        for band_name, (fmin, fmax) in freq_bands.items():
            fmask = (freqs >= fmin) & (freqs <= fmax)
            if not np.any(fmask):
                continue
            A_k = A_bands_est[band_name]
            W_band = np.zeros_like(W_pred_j[fmask, :], dtype=np.complex128)
            for i, var_i in enumerate(variables):
                W_band += A_k[i, j] * cwt_results[var_i]["W"][fmask, :]
            W_pred_j[fmask, :] = W_band

        x_pred_j = wavelet.icwt(W_pred_j, scales, dt, dj, mother)
        # icwt should be real for real inputs; keep robust
        X_pred[j] = safe_array(np.real(x_pred_j))

    return X_pred


# ==========================================================
# DYNOTEARS-lite: dynamic NOTEARS optimization with L-BFGS-B
# ==========================================================

@dataclass
class DynotearsConfig:
    maxlag: int = 3
    lambda1_A: float = 0.01
    lambda1_B: float = 0.01
    rho: float = 1.0
    alpha: float = 0.0
    max_iter: int = 10
    h_tol: float = 1e-8
    rho_max: float = 1e16
    w_threshold: float = 0.3


def _build_var_design(X: np.ndarray, maxlag: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return Y (T-maxlag,d) and Z (T-maxlag, d*maxlag) with stacked lags."""
    T, d = X.shape
    Y = X[maxlag:, :]
    Z_blocks = []
    for l in range(1, maxlag + 1):
        Z_blocks.append(X[maxlag - l:T - l, :])
    Z = np.concatenate(Z_blocks, axis=1)
    return Y, Z


def dynotears_fit(X: np.ndarray, cfg: DynotearsConfig,
                  A0: np.ndarray | None = None,
                  B0: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Fit instantaneous A (dxd) and lagged B (d*L x d) using augmented Lagrangian.

    Model: Y \approx Y A + Z B
    where Y = X[maxlag:], Z stacks lagged X.
    """
    X = np.asarray(X, dtype=float)
    T, d = X.shape
    Y, Z = _build_var_design(X, cfg.maxlag)
    n = Y.shape[0]

    if A0 is None:
        A0 = np.zeros((d, d), dtype=float)
    if B0 is None:
        B0 = np.zeros((d * cfg.maxlag, d), dtype=float)

    # parameter vector: vec(A) then vec(B)
    x0 = np.concatenate([A0.reshape(-1), B0.reshape(-1)])

    # bounds enforce diag(A)=0
    bounds = []
    for i in range(d):
        for j in range(d):
            if i == j:
                bounds.append((0.0, 0.0))
            else:
                bounds.append((None, None))
    bounds += [(None, None)] * (d * cfg.maxlag * d)

    rho = cfg.rho
    alpha = cfg.alpha
    h_prev = np.inf

    def unpack(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        A = x[: d * d].reshape(d, d)
        B = x[d * d:].reshape(d * cfg.maxlag, d)
        return A, B

    def objective(x: np.ndarray) -> float:
        A, B = unpack(x)
        Y_hat = Y @ A + Z @ B
        resid = Y - Y_hat
        loss = 0.5 / n * float(np.sum(resid ** 2))

        hA = h_func(A)
        penalty = 0.5 * rho * (hA ** 2) + alpha * hA

        l1 = cfg.lambda1_A * float(np.sum(np.abs(A))) + cfg.lambda1_B * float(np.sum(np.abs(B)))
        return loss + penalty + l1

    def grad(x: np.ndarray) -> np.ndarray:
        # finite-diff is too slow; use analytic gradients for least squares + numeric for h(A)
        # We keep it simple and stable: analytic for LS, numeric for h(A) + l1 subgrad.
        A, B = unpack(x)

        Y_hat = Y @ A + Z @ B
        resid = (Y_hat - Y)  # [n,d]

        # d/dA  0.5/n ||Y - Y_hat||^2 = (1/n) Y^T (Y_hat - Y)
        gA = (Y.T @ resid) / n
        gB = (Z.T @ resid) / n

        # l1 subgrad (0 at 0)
        gA += cfg.lambda1_A * np.sign(A)
        gB += cfg.lambda1_B * np.sign(B)

        # acyclicity gradient: dh/dA = 2 * (expm(A∘A)^T) ∘ A
        E = expm(A * A)
        g_h = (E.T) * (2 * A)
        hA = float(np.trace(E) - d)
        gA += (rho * hA + alpha) * g_h

        # enforce diag grad 0 (since diag is fixed)
        np.fill_diagonal(gA, 0.0)

        return np.concatenate([gA.reshape(-1), gB.reshape(-1)])

    for _ in range(cfg.max_iter):
        res = minimize(
            fun=objective,
            x0=x0,
            jac=grad,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-8},
        )

        x0 = res.x
        A, B = unpack(x0)
        hA = h_func(A)

        if hA <= cfg.h_tol:
            break

        # update rho if not decreasing enough
        if hA > 0.25 * h_prev and rho < cfg.rho_max:
            rho *= 10.0
        alpha += rho * hA
        h_prev = hA

        if rho >= cfg.rho_max:
            break

    A, B = unpack(x0)
    return A, B


# ==========================================================
# Benchmark runner
# ==========================================================

@dataclass
class BenchmarkConfig:
    duration: float = 20
    fs: int = 500
    d: int = 3
    noise_std: float = 0.1
    n_runs: int = 10
    seed0: int = 123
    w_threshold: float = 0.3


def run_one(seed: int,
            freq_bands: Dict[str, Tuple[float, float]],
            dyn_cfg: DynotearsConfig,
            bench_cfg: BenchmarkConfig) -> Dict[str, float]:
    set_seed(seed)
    rng = np.random.default_rng(seed)

    data, t, X_total, A_true_bands = generate_data_simple(
        duration=bench_cfg.duration,
        fs=bench_cfg.fs,
        d=bench_cfg.d,
        band_ranges=freq_bands,
        noise_std=bench_cfg.noise_std,
        rng=rng,
    )

    variables = [f"x{i + 1}" for i in range(bench_cfg.d)]

    # fit DYNOTEARS-lite on time-domain signals (single-scale)
    X_mat = np.stack([data[v] for v in variables], axis=1)  # [T,d]

    A_est, B_est_lag = dynotears_fit(X_mat, dyn_cfg)

    # Build an "equivalent" single adjacency (use |A_est|; replicate across bands)
    A_single = np.abs(A_est)
    np.fill_diagonal(A_single, 0.0)

    A_est_bands = {bn: A_single.copy() for bn in freq_bands.keys()}

    # wavelet reconstruction (full-series)
    dt = 1.0 / bench_cfg.fs
    mother = wavelet.Morlet(6)
    s0 = 2 * dt
    dj = 1 / 12
    J = int(7 / dj)
    # Reconstruction / prediction (single-scale):
    # DYNOTEARS fits: x_t = x_t A + z_t B + eps  =>  x_t = z_t B (I - A)^{-1}
    X_td = X_total.T  # [T, d]
    L = dyn_cfg.maxlag
    if L >= X_td.shape[0]:
        raise ValueError(f"lag L={L} is too large for T={X_td.shape[0]}")
    Y = X_td[L:, :]  # [T-L, d]
    Z = np.hstack([X_td[L - l: -l, :] for l in range(1, L + 1)])  # [T-L, d*L]

    # --- reconstruction (single-scale) ---
    d = X_td.shape[1]  # 节点数
    A_inst = A_est  # 即时矩阵
    B_lag = B_est_lag  # 滞后矩阵

    I = np.eye(d)
    IA = I - A_inst
    IA = IA + 1e-6 * I  # ridge for numerical stability
    try:
        IA_inv = np.linalg.inv(IA)
    except np.linalg.LinAlgError:
        IA_inv = np.linalg.pinv(IA)

    Y_pred = (Z @ B_lag) @ IA_inv  # [T-L, d]
    X_true = Y.T
    X_pred = Y_pred.T


    # metrics: structure per band (replicated) and reconstruction (global)
    f1s, shds, tprs = [], [], []
    for bn in freq_bands.keys():
        B_true = binarize(A_true_bands[bn], thr=0.0)  # true edges are nonzero
        B_hat = binarize(A_single, thr=bench_cfg.w_threshold)
        f1s.append(f1_from_adj(B_true, B_hat))
        shd, tpr = shd_tpr_from_adj(B_true, B_hat)
        shds.append(shd)
        tprs.append(tpr)

    mse, mae = mse_mae_standardized(X_true, X_pred)

    return {
        "seed": seed,
        "F1": float(np.mean(f1s)),
        "SHD": float(np.mean(shds)),
        "TPR": float(np.mean(tprs)),
        "MSE": mse,
        "MAE": mae,
        "h(A)": h_func(A_est),
    }


def run_benchmark(freq_bands: Dict[str, Tuple[float, float]] | None = None,
                  dyn_cfg: DynotearsConfig | None = None,
                  bench_cfg: BenchmarkConfig | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if freq_bands is None:
        freq_bands = {"Low": (0.1, 6), "Mid": (7, 15), "High": (17, 25)}
    if dyn_cfg is None:
        dyn_cfg = DynotearsConfig()
    if bench_cfg is None:
        bench_cfg = BenchmarkConfig()

    rows = []
    for r in range(bench_cfg.n_runs):
        seed = bench_cfg.seed0 + r
        rows.append(run_one(seed, freq_bands=freq_bands, dyn_cfg=dyn_cfg, bench_cfg=bench_cfg))

    df = pd.DataFrame(rows)
    summary = df[["F1", "SHD", "TPR", "MSE", "MAE", "h(A)"]].agg(["mean", "std"]).T
    return df, summary


def main():
    # frequency bands consistent with CP-FCD
    freq_bands = {"Low": (0.1, 6), "Mid": (7, 15), "High": (17, 25)}

    dyn_cfg = DynotearsConfig(
        maxlag=3,
        lambda1_A=0.01,
        lambda1_B=0.01,
        max_iter=10,
        h_tol=1e-8,
        w_threshold=0.3,
    )

    bench_cfg = BenchmarkConfig(
        duration=20,
        fs=500,
        d=3,
        noise_std=0.1,
        n_runs=30,
        seed0=123,
        w_threshold=0.3,
    )

    df, summary = run_benchmark(freq_bands=freq_bands, dyn_cfg=dyn_cfg, bench_cfg=bench_cfg)

    print("\nPer-run metrics:")
    print(df)
    print("\nSummary (mean/std):")
    print(summary)

    out_csv = "dynotears_benchmark_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
