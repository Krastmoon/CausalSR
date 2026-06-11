# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


# ===================== metrics =====================
def count_accuracy_no_dag_check(B_true, B_est):
    B_true = (np.asarray(B_true) != 0).astype(int)
    B_est = (np.asarray(B_est) != 0).astype(int)
    d = B_true.shape[0]

    pred = np.flatnonzero(B_est)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])

    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)

    pred_size = len(pred)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)

    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)

    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = int(len(extra_lower) + len(missing_lower) + len(reverse))

    return {"fdr": fdr, "tpr": tpr, "fpr": fpr, "shd": shd, "nnz": pred_size}


def f1_from_fdr_tpr(acc_dict):
    precision = 1.0 - acc_dict["fdr"]
    recall = acc_dict["tpr"]
    return 0.0 if (precision + recall) <= 0 else (2 * precision * recall / (precision + recall))


# ===================== data generation (self-contained; same logic) =====================
def sample_random_dag(d=3, p_edge=0.3, w_min=0.7, w_max=0.9, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    while True:
        order = rng.permutation(d)
        A = np.zeros((d, d), dtype=float)
        for u_rank in range(d):
            for v_rank in range(u_rank + 1, d):
                u, v = order[u_rank], order[v_rank]
                if rng.random() < p_edge:
                    A[u, v] = rng.uniform(w_min, w_max)
        if np.any(A != 0):
            return A


def roll_shift(x, lag, dt=1.0, mode="zero"):
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
    return np.fft.irfft(Xf * phase_shift, n=n)


def _band_limited_noise(center_hz, bw_hz, fs, T, rng):
    freqs = np.fft.rfftfreq(T, 1 / fs)
    sigma = bw_hz / 2.355 if bw_hz > 0 else (center_hz / 10 + 1e-6)
    win = np.exp(-0.5 * ((freqs - center_hz) / (sigma + 1e-12)) ** 2)

    phase = rng.uniform(0, 2 * np.pi, size=len(freqs))
    amp = rng.normal(0, 1, size=len(freqs))
    X = win * amp * (np.cos(phase) + 1j * np.sin(phase))
    x = np.fft.irfft(X, n=T)
    if np.std(x) > 1e-12:
        x /= np.std(x)
    return x


def synth_base_complex(d, T, fs, freqs, rng, bandwidth_ratio=0.15):
    t = np.arange(T) / fs
    Z = np.zeros((d, T), float)
    for i in range(d):
        f0 = float(freqs[i])
        bw = max(f0 * bandwidth_ratio, fs / T)
        nb = _band_limited_noise(f0, bw, fs, T, rng)
        carrier = np.sin(2 * np.pi * f0 * t)
        z = 0.7 * nb + 0.3 * carrier + rng.normal(0, 0.02, size=T)
        if np.std(z) > 1e-12:
            z /= np.std(z)
        Z[i] = z
    return Z


def generate_freqs_dict(d, band_ranges, rng, margin=0.5):
    freqs_dict = {}
    for band, (fmin, fmax) in band_ranges.items():
        lo, hi = fmin + margin, fmax - margin
        if hi <= lo:
            raise ValueError(f"[{band}] invalid band after margin")
        freqs = rng.uniform(lo, hi, size=d)
        freqs_dict[band] = list(np.sort(freqs))
    return freqs_dict


def generate_data_simple(duration=20, fs=500, d=5, band_ranges=None, noise_std=0.1, seed=0):
    if band_ranges is None:
        band_ranges = {"Low": (0.1, 6), "Mid": (7, 15), "High": (17, 25)}

    rng = np.random.default_rng(seed)
    T = int(duration * fs)
    dt = 1.0 / fs

    freqs_dict = generate_freqs_dict(d, band_ranges, rng=rng, margin=0.5)

    A_bands, tau_bands, comps = {}, {}, {}
    X_total = np.zeros((d, T), dtype=float)

    for band in band_ranges.keys():
        A = sample_random_dag(d=d, rng=rng)
        A_bands[band] = A

        f = np.array(freqs_dict[band], dtype=float)
        Z = synth_base_complex(d=d, T=T, fs=fs, freqs=f, rng=rng)

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
        for i in range(d):
            for j in range(d):
                if i != j and A[i, j] != 0:
                    X_band[j] += A[i, j] * roll_shift(Z[i], tau[i, j], dt=dt, mode="zero")

        comps[band] = X_band
        X_total += X_band

    X_total += rng.normal(0, noise_std, size=X_total.shape)
    return X_total, A_bands


# ===================== FIXED Granger PW -> summary adjacency =====================
def granger_pw_summary_adj(X_df, sig_level=0.05, maxlag=5, verbose=False, test="ssr_ftest"):
    """
    Returns B (d,d): B[i,j]=1 means i -> j

    FIX: treat BOTH 1 and 2 as edges (since 2 is the original "significant" mark).
    """
    names = list(X_df.columns)
    d = len(names)
    dataset = pd.DataFrame(np.zeros((d, d), dtype=int), columns=names, index=names)

    # mark significant: dataset.loc[c, r] = 2 (c causes r)
    for c in names:
        for r in names:
            if r == c:
                continue
            try:
                test_result = grangercausalitytests(X_df[[r, c]], maxlag=maxlag, verbose=verbose)
                p_values = [float(test_result[i + 1][0][test][1]) for i in range(maxlag)]
                if np.min(p_values) < sig_level:
                    dataset.loc[c, r] = 2
            except Exception:
                # if a pair fails due to numerical issues, just skip it
                continue

    # keep the same post-process style as your file (optional)
    for c in names:
        for r in names:
            if dataset.loc[c, r] == 2 and dataset.loc[r, c] == 0:
                dataset.loc[r, c] = 1
            if r == c:
                dataset.loc[r, c] = 1

    # === KEY FIX: build adjacency from "2" as well ===
    name_to_idx = {n: i for i, n in enumerate(names)}
    B = np.zeros((d, d), dtype=int)

    # dataset.loc[c, r] == 2 means c -> r
    for c in names:
        for r in names:
            if c == r:
                continue
            if dataset.loc[c, r] == 2:
                B[name_to_idx[c], name_to_idx[r]] = 1

    # dataset.loc[r, c] == 1 also indicates c -> r (one-way case)
    for r in names:
        for c in names:
            if r == c:
                continue
            if dataset.loc[r, c] == 1:
                B[name_to_idx[c], name_to_idx[r]] = 1

    np.fill_diagonal(B, 0)
    return B


def evaluate_summary_vs_band_truth(A_bands, B_est_summary, band_ranges):
    per_band = {}
    f1s, tprs, shds = [], [], []
    for band in band_ranges.keys():
        B_true = (np.asarray(A_bands[band]) != 0).astype(int)
        acc = count_accuracy_no_dag_check(B_true, B_est_summary)
        F1 = f1_from_fdr_tpr(acc)

        per_band[band] = {
            "F1": float(F1),
            "TPR": float(acc["tpr"]),
            "SHD": float(acc["shd"]),
            "FDR": float(acc["fdr"]),
            "NNZ": int(acc["nnz"]),
        }
        f1s.append(F1)
        tprs.append(acc["tpr"])
        shds.append(acc["shd"])
    avg = {"F1": float(np.mean(f1s)), "TPR": float(np.mean(tprs)), "SHD": float(np.mean(shds))}
    return per_band, avg


# ===================== main =====================
def main():
    N_RUNS = 10
    duration = 20
    fs = 500
    d = 50
    noise_std = 0.1

    band_ranges = {"Low": (0.1, 6), "Mid": (7, 15), "High": (17, 25)}

    sig_level = 0.0000000001
    maxlag = 3
    verbose = False

    rows = []
    for r in range(N_RUNS):
        seed = 1234 + r

        X_total, A_bands = generate_data_simple(
            duration=duration, fs=fs, d=d, band_ranges=band_ranges, noise_std=noise_std, seed=seed
        )
        var_names = [f"x{i+1}" for i in range(d)]
        X_df = pd.DataFrame(X_total.T, columns=var_names)

        B_est_summary = granger_pw_summary_adj(
            X_df=X_df, sig_level=sig_level, maxlag=maxlag, verbose=verbose, test="ssr_ftest"
        )

        per_band, avg = evaluate_summary_vs_band_truth(A_bands, B_est_summary, band_ranges)

        rows.append({"Seed": seed, **avg})

        print(f"\n[seed={seed}] AVG over bands: {avg}")
        for bn, m in per_band.items():
            print(f"  - {bn}: {m}")

    df = pd.DataFrame(rows)
    print("\n================ Per-run AVG (over bands) ================")
    print(df)

    print("\n================ Summary (mean/std over runs) ================")
    print(df[["F1", "TPR", "SHD"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
