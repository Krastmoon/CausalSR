import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# =========================
# 1) 你的合成数据生成（保留你原来的逻辑：三频段各自 DAG -> 混合成总时间序列）
# =========================
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
    Yf = Xf * phase_shift
    y = np.fft.irfft(Yf, n=n)
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
                       bandwidth_ratio=0.15,
                       target_abs_corr=0.25, max_decor_iter=3):
    rng = np.random.default_rng() if rng is None else rng
    t = np.arange(T) / fs
    Z = np.zeros((d, T), float)

    for i in range(d):
        f0 = float(freqs[i])
        a = rng.uniform(*amp_range)

        if mode == "narrowband_noise":
            bw = max(f0 * bandwidth_ratio, fs / T)
            nb = _band_limited_noise(f0, bw, fs, T, rng)
            carrier = np.sin(2 * np.pi * f0 * t + phi_shared)
            z = a * (0.7 * nb + 0.3 * carrier)
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
                         band_ranges={"Low": (0.1, 6), "Mid": (7, 15), "High": (17, 25)},
                         noise_std=0.1, rng=None,
                         base_mode="narrowband_noise",
                         phi_shared_map=None,
                         show=False):
    rng = np.random.default_rng() if rng is None else rng
    T = int(duration * fs)
    dt = 1.0 / fs
    t = np.arange(T) / fs

    freqs_dict = {"Low": [2, 3, 4], "Mid": [11, 12, 13], "High": [21, 22, 23]}
    if phi_shared_map is None:
        phi_shared_map = {k: 0.0 for k in band_ranges}

    A_bands, tau_bands, comps = {}, {}, {}
    X_total = np.zeros((d, T))

    for band, (_f_low, _f_high) in band_ranges.items():
        A = sample_random_dag(d, rng=rng)
        A_bands[band] = A

        f = np.array(freqs_dict[band], dtype=float)

        Z = synth_base_complex(d, T, fs, f,
                               phi_shared=phi_shared_map[band],
                               rng=rng,
                               mode=base_mode)

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
                    lag = tau[i, j]
                    X_band[j] += A[i, j] * roll_shift(Z[i], lag, dt=dt, mode="zero")

        comps[band] = X_band
        X_total += X_band

        if show:
            print(f"\n==== {band} band ====")
            print("A (i->j):\n", A)
            print("freqs (Hz):", f)
            print("tau (samples, may be float):\n", np.round(tau, 2))

    X_total += rng.normal(0, noise_std, size=X_total.shape)
    data = {f"x{i+1}": X_total[i] for i in range(d)}
    return data, t, X_total, A_bands, tau_bands, comps


# =========================
# 2) 结构指标（不强制 DAG；兼容 PCMCI 可能产生环）
# =========================
def binarize_adj(A, thr=1e-12):
    A = np.array(A)
    B = (np.abs(A) > thr).astype(int)
    np.fill_diagonal(B, 0)
    return B

def shd_tpr_f1(B_true, B_est):
    """
    B_true, B_est: {0,1} adjacency, directed.
    - reverse: pred i->j but true j->i
    - extra: pred edge not in skeleton(true)
    - missing: true skeleton edge not predicted in skeleton
    """
    d = B_true.shape[0]
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true == 1)
    cond_rev = np.flatnonzero(B_true.T == 1)
    cond_skel = np.unique(np.concatenate([cond, cond_rev]))

    # TP: correct direction
    tp = np.intersect1d(pred, cond, assume_unique=False)
    # reverse: predicted but opposite of true
    extra = np.setdiff1d(pred, cond, assume_unique=False)
    rev = np.intersect1d(extra, cond_rev, assume_unique=False)

    # FP: predicted edges not in skeleton
    fp = np.setdiff1d(pred, cond_skel, assume_unique=False)

    pred_size = len(pred)
    cond_size = len(cond)

    tpr = float(len(tp)) / max(cond_size, 1)
    precision = float(len(tp)) / max(pred_size, 1)
    f1 = 0.0 if (precision + tpr) == 0 else 2 * precision * tpr / (precision + tpr)

    # SHD: extra_undirected + missing_undirected + reverse
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=False)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=False)
    shd = len(extra_lower) + len(missing_lower) + len(rev)

    return {"F1": f1, "TPR": tpr, "SHD": float(shd)}


# =========================
# 3) 标准化后的 MSE / MAE（对每个变量 z-score，再算整体误差）
# =========================
def zscore(X, eps=1e-12):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.maximum(sd, eps)
    return (X - mu) / sd

def mse_mae_std(X_true, X_pred):
    Xt = zscore(X_true)
    Xp = zscore(X_pred)
    mse = np.mean((Xt - Xp) ** 2)
    mae = np.mean(np.abs(Xt - Xp))
    return {"MSE_std": float(mse), "MAE_std": float(mae)}


# =========================
# 4) PCMCI：用 tigramite
#    输出：A_est (d,d) 二值（只要任意 lag 显著就算 i->j）
#          links: 详细 lag 信息
# =========================
def run_pcmci(X, tau_max=10, alpha_level=0.05, cond_ind_test="ParCorr"):
    """
    X: shape [d, T]  (注意 tigramite 需要 [T, d])
    """
    from tigramite.data_processing import DataFrame as TigDataFrame
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr

    X_Td = X.T  # [T, d]
    dataframe = TigDataFrame(X_Td)

    if cond_ind_test == "ParCorr":
        cit = ParCorr(significance='analytic')
    else:
        raise ValueError("Only ParCorr is configured in this script.")

    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cit, verbosity=0)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=None)

    p_matrix = results["p_matrix"]  # [d, d, tau_max+1]
    # 只看 lag>=1
    sig = (p_matrix[:, :, 1:] < alpha_level)

    d = X.shape[0]
    A_est = np.zeros((d, d), dtype=int)
    # i -> j if any lag significant
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if np.any(sig[i, j, :]):
                A_est[i, j] = 1
    return A_est, results


# =========================
# 5) 用 PCMCI 选出的父集做线性回归重构（用于 MSE/MAE）
#    这里做“每个变量 j 的 1-step 预测”：t 时刻用父节点的 t-lag 值线性拟合
# =========================
def reconstruct_via_selected_lags(X, pcmci_results, alpha_level=0.05, tau_max=10):
    """
    X: [d, T]
    pcmci_results: results dict from tigramite
    输出 X_pred: [d, T]，前 tau_max 步会是 0（或你可选择丢弃）
    """
    p_matrix = pcmci_results["p_matrix"]  # [d,d,tau_max+1]
    val_matrix = pcmci_results["val_matrix"]  # partial corr values (same shape)
    d, T = X.shape

    X_pred = np.zeros_like(X)
    # 从 t=tau_max 开始预测
    for j in range(d):
        # 收集显著父项 (i, lag)
        parents = []
        for i in range(d):
            if i == j:
                continue
            for lag in range(1, tau_max + 1):
                if p_matrix[i, j, lag] < alpha_level:
                    parents.append((i, lag))

        if len(parents) == 0:
            continue

        # 构造回归：y = X[j, tau_max:]，features = X[i, tau_max-lag: T-lag]
        y = X[j, tau_max:]
        F = []
        for (i, lag) in parents:
            F.append(X[i, tau_max - lag: T - lag])
        F = np.stack(F, axis=1)  # [T-tau_max, num_parents]

        # 最小二乘
        # w = argmin ||F w - y||
        w, *_ = np.linalg.lstsq(F, y, rcond=None)

        yhat = F @ w
        X_pred[j, tau_max:] = yhat

    return X_pred


# =========================
# 6) 多轮实验：输出表格 + 汇总均值/方差
# =========================
def run_benchmark(
    n_runs=30,
    seed0=0,
    duration=20,
    fs=500,
    d=3,
    tau_max=10,
    alpha_level=0.05,
    noise_std=0.1
):
    band_ranges = {"Low": (0.1, 6), "Mid": (7, 15), "High": (17, 25)}
    variables = [f"x{i+1}" for i in range(d)]

    rows = []
    for r in range(n_runs):
        rng = np.random.default_rng(seed0 + r)

        data, t, X_total, A_bands_true, _, _ = generate_data_simple(
            duration=duration, fs=fs, d=d,
            band_ranges=band_ranges,
            noise_std=noise_std,
            rng=rng,
            base_mode="narrowband_noise",
            show=False
        )

        # 1) PCMCI on total time series
        A_pcmci, pcmci_results = run_pcmci(X_total, tau_max=tau_max, alpha_level=alpha_level)

        # 2) 结构指标：对每个频段真值分别算（PCMCI 图复用）
        for band in ["Low", "Mid", "High"]:
            B_true = binarize_adj(A_bands_true[band])
            B_est = A_pcmci.copy()

            m_struct = shd_tpr_f1(B_true, B_est)

            # 3) 重构误差：用 PCMCI 选出的 lag 父集做线性 1-step 重构（在 total 上算一次即可）
            X_pred = reconstruct_via_selected_lags(X_total, pcmci_results,
                                                   alpha_level=alpha_level, tau_max=tau_max)
            m_recon = mse_mae_std(X_total[:, tau_max:], X_pred[:, tau_max:])

            rows.append({
                "run": r,
                "band": band,
                **m_struct,
                **m_recon,
                "edges_true": int(B_true.sum()),
                "edges_est": int(B_est.sum())
            })

    df = pd.DataFrame(rows)

    # 汇总（按 band）
    summary_band = df.groupby("band")[["F1", "TPR", "SHD", "MSE_std", "MAE_std"]].agg(["mean", "std"])
    # 汇总（整体）
    summary_all = df[["F1", "TPR", "SHD", "MSE_std", "MAE_std"]].agg(["mean", "std"])

    return df, summary_band, summary_all


def main():
    df, summary_band, summary_all = run_benchmark(
        n_runs=10,
        seed0=0,
        duration=20,
        fs=500,
        d=3,
        tau_max=10,
        alpha_level=0.05,
        noise_std=0.1
    )

    print("\n===== Per-run results =====")
    print(df)

    print("\n===== Summary by band (mean/std) =====")
    print(summary_band)

    print("\n===== Summary overall (mean/std) =====")
    print(summary_all)

    # 保存表格
    df.to_csv("pcmci_benchmark_runs.csv", index=False)
    summary_band.to_csv("pcmci_benchmark_summary_by_band.csv")
    summary_all.to_csv("pcmci_benchmark_summary_overall.csv")
    print("\nSaved:")
    print(" - pcmci_benchmark_runs.csv")
    print(" - pcmci_benchmark_summary_by_band.csv")
    print(" - pcmci_benchmark_summary_overall.csv")


if __name__ == "__main__":
    main()
