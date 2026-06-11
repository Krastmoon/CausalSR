import numpy as np
import pandas as pd

# ======= tigramite / pcmciplus =======
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import parcorr, cmiknn
from tigramite import data_processing as pp

# =========================
# 1) Synthetic data generator (与你原来思路一致的简化版)
#    - 三个频段各自一个 DAG + 非整数滞后混合
#    - 最终 X_total = sum_k X_band^k + noise
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

    # fractional delay via frequency-domain phase shift
    tau = lag * dt
    Xf = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, dt)
    phase_shift = np.exp(-2j * np.pi * freqs * tau)
    Yf = Xf * phase_shift
    y = np.fft.irfft(Yf, n=n)
    return y

def _band_limited_noise(center_hz, bw_hz, fs, T, rng):
    n = T
    freqs = np.fft.rfftfreq(n, 1/fs)
    sigma = bw_hz/2.355 if bw_hz > 0 else (center_hz/10 + 1e-6)
    win = np.exp(-0.5*((freqs-center_hz)/(sigma+1e-12))**2)
    phase = rng.uniform(0, 2*np.pi, size=len(freqs))
    amp = rng.normal(0, 1, size=len(freqs))
    X = win * amp * (np.cos(phase) + 1j*np.sin(phase))
    x = np.fft.irfft(X, n=n)
    if np.std(x) > 1e-12:
        x /= np.std(x)
    return x

def synth_base_complex(d, T, fs, freqs, rng, bandwidth_ratio=0.15):
    t = np.arange(T)/fs
    Z = np.zeros((d, T), float)
    for i in range(d):
        f0 = float(freqs[i])
        bw = max(f0 * bandwidth_ratio, fs/T)
        nb = _band_limited_noise(f0, bw, fs, T, rng)
        carrier = np.sin(2*np.pi*f0*t)
        z = (0.7*nb + 0.3*carrier) + rng.normal(0, 0.02, size=T)
        std = np.std(z)
        if std > 1e-12:
            z /= std
        Z[i] = z
    return Z

def generate_data_simple(duration=20, fs=500, d=3,
                         band_ranges={"Low": (0.1, 6), "Mid": (7, 15), "High": (17, 25)},
                         noise_std=0.1, p_edge=0.3, rng=None, show=False):
    rng = np.random.default_rng() if rng is None else rng
    T = int(duration * fs)
    dt = 1.0/fs
    t = np.arange(T)/fs

    freqs_dict = {"Low": [2, 3, 4], "Mid": [11, 12, 13], "High": [21, 22, 23]}
    X_total = np.zeros((d, T))
    A_bands, tau_bands = {}, {}

    for band in band_ranges.keys():
        A = sample_random_dag(d, p_edge=p_edge, rng=rng)
        A_bands[band] = (A != 0).astype(int)  # ground-truth binary graph for metrics

        f = np.array(freqs_dict[band], dtype=float)
        Z = synth_base_complex(d, T, fs, f, rng=rng)

        tau = np.zeros((d, d), dtype=float)
        X_band = Z.copy()
        for i in range(d):
            max_tau = fs/f[i]/8.0
            for j in range(d):
                if i == j or A[i, j] == 0:
                    continue
                tau[i, j] = rng.uniform(1.0, max(1.0, max_tau))
                X_band[j] += A[i, j] * roll_shift(Z[i], tau[i, j], dt=dt, mode="zero")
        tau_bands[band] = tau

        X_total += X_band

        if show:
            print(f"\n==== {band} ====")
            print("A true (binary):\n", A_bands[band])
            print("tau(samples):\n", np.round(tau, 2))

    X_total += rng.normal(0, noise_std, size=X_total.shape)

    data = {f"x{i+1}": X_total[i] for i in range(d)}
    df = pd.DataFrame(data)
    return df, A_bands


# =========================
# 2) PCMCI+ runner -> estimated adjacency (collapse over lags)
#    逻辑参考你上传的 PCMCIplus.py :contentReference[oaicite:1]{index=1}
# =========================

def run_pcmciplus_get_adj(data_df, tau_max=5, cond_ind_test="ParCorr", alpha=0.05, verbosity=0):
    if cond_ind_test == "CMIknn":
        cit = cmiknn.CMIknn()
    else:
        cit = parcorr.ParCorr()

    tig_df = pp.DataFrame(data_df.values, var_names=list(data_df.columns))

    pcmci = PCMCI(dataframe=tig_df, cond_ind_test=cit, verbosity=verbosity)
    res = pcmci.run_pcmciplus(
        selected_links=None,
        tau_min=0,
        tau_max=tau_max,
        pc_alpha=alpha,
        contemp_collider_rule='majority',
        conflict_resolution=True,
        reset_lagged_links=False,
        fdr_method='none',
    )

    graph = res["graph"]        # shape [d, d, tau_max+1], strings like "-->", "<--", etc.
    valm  = res["val_matrix"]   # same shape, strength

    d = data_df.shape[1]
    B_est = np.zeros((d, d), dtype=int)

    # 折叠：只要存在任何 lag 的显著有向边 i -> j，就记为 1
    # 这里用 graph 非空且不是 "<--" 的方向判定方式，延续你文件里的做法 :contentReference[oaicite:2]{index=2}
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            # tau index: 0..tau_max
            for tau in range(graph.shape[2]):
                g = graph[i, j, tau]
                if g is None:
                    continue
                if (g != "") and (g != "<--"):
                    B_est[i, j] = 1
                    break

    return B_est, res


# =========================
# 3) Metrics: F1 / TPR / SHD  +  standardized MSE/MAE
# =========================

def f1_tpr_shd(B_true, B_est):
    """
    B_true, B_est: binary adjacency [d,d], directed
    """
    d = B_true.shape[0]

    # Directed edges excluding diagonal
    mask = ~np.eye(d, dtype=bool)

    TPs = np.sum((B_true == 1) & (B_est == 1) & mask)
    FPs = np.sum((B_true == 0) & (B_est == 1) & mask)
    FNs = np.sum((B_true == 1) & (B_est == 0) & mask)

    precision = TPs / max(TPs + FPs, 1)
    recall    = TPs / max(TPs + FNs, 1)   # == TPR
    f1        = 2 * precision * recall / max(precision + recall, 1e-12)

    # SHD (simple directed version):
    # extra edges + missing edges + reversed edges (count reversed as 1 each)
    # We'll compute "skeleton" mismatch + direction mismatch:
    Bu_true = ((B_true + B_true.T) > 0).astype(int)
    Bu_est  = ((B_est  + B_est.T ) > 0).astype(int)

    extra   = np.sum((Bu_est == 1) & (Bu_true == 0)) // 2
    missing = np.sum((Bu_true == 1) & (Bu_est == 0)) // 2

    # reversed: edges that exist in both skeletons but direction opposite
    reversed_cnt = 0
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if B_true[i, j] == 1 and B_est[j, i] == 1 and B_est[i, j] == 0:
                reversed_cnt += 1
    # each reversed pair counted once because only one direction present in B_true for a DAG-like truth
    shd = int(extra + missing + reversed_cnt)

    return float(f1), float(recall), float(shd)


def standardized_mse_mae(X_true, X_pred):
    """
    X_true, X_pred: shape [T, d] or [d, T]
    Standardize per variable using X_true mean/std, then compute MSE/MAE averaged over variables.
    """
    if X_true.shape[0] < X_true.shape[1]:
        # assume [d,T] -> [T,d]
        X_true = X_true.T
        X_pred = X_pred.T

    mu = X_true.mean(axis=0, keepdims=True)
    sd = X_true.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)

    Z_true = (X_true - mu) / sd
    Z_pred = (X_pred - mu) / sd

    mse = np.mean((Z_true - Z_pred)**2)
    mae = np.mean(np.abs(Z_true - Z_pred))
    return float(mse), float(mae)


# =========================
# 4) Benchmark loop: multi-runs + summary table
# =========================

def run_pcmciplus_benchmark(
    n_runs=20,
    seed0=0,
    duration=20,
    fs=500,
    d=3,
    tau_max=5,
    cond_ind_test="ParCorr",
    alpha=0.05,
    noise_std=0.1,
    p_edge=0.3,
):
    rows = []

    for r in range(n_runs):
        rng = np.random.default_rng(seed0 + r)

        data_df, A_true_bands = generate_data_simple(
            duration=duration, fs=fs, d=d,
            noise_std=noise_std, p_edge=p_edge,
            rng=rng, show=False
        )

        # PCMCI+ estimate (single-scale)
        B_est, _ = run_pcmciplus_get_adj(
            data_df, tau_max=tau_max,
            cond_ind_test=cond_ind_test,
            alpha=alpha,
            verbosity=0
        )

        # Reconstruction metrics:
        # PCMCI+ 本身不做重构，为了对齐你的“结构+重构”评估，
        # 这里给一个最朴素的线性 one-step 重构：x(t) <- sum_i B_est[i,j] x_i(t-1)
        # （你如果想换成 VAR(p) 回归重构，我也可以给你替换版本）
        X = data_df.values  # [T,d]
        T = X.shape[0]
        X_pred = np.zeros_like(X)
        X_pred[0] = X[0]
        for t in range(1, T):
            for j in range(d):
                parents = np.where(B_est[:, j] == 1)[0]
                if len(parents) == 0:
                    X_pred[t, j] = X[t-1, j]  # no parent -> persistence baseline
                else:
                    X_pred[t, j] = np.mean(X[t-1, parents])  # simple average influence

        mse_z, mae_z = standardized_mse_mae(X_true=X, X_pred=X_pred)

        # Structure metrics per band: treat same B_est for all bands (你说的做法)
        for band_name in ["Low", "Mid", "High"]:
            B_true = A_true_bands[band_name]
            f1, tpr, shd = f1_tpr_shd(B_true, B_est)

            rows.append({
                "run": r,
                "band": band_name,
                "F1": f1,
                "TPR": tpr,
                "SHD": shd,
                "MSE(z)": mse_z,
                "MAE(z)": mae_z,
                "nnz_est": int(B_est.sum()),
                "nnz_true": int(B_true.sum()),
            })

    df = pd.DataFrame(rows)

    # summary per band
    summary = df.groupby("band")[["F1", "TPR", "SHD", "MSE(z)", "MAE(z)"]].agg(["mean", "std"]).reset_index()

    # overall summary (all bands pooled)
    summary_all = df[["F1", "TPR", "SHD", "MSE(z)", "MAE(z)"]].agg(["mean", "std"])

    return df, summary, summary_all


def main():
    df, summary, summary_all = run_pcmciplus_benchmark(
        n_runs=30,
        seed0=0,
        duration=20,
        fs=500,
        d=3,
        tau_max=5,
        cond_ind_test="ParCorr",  # or "CMIknn"
        alpha=0.05,
        noise_std=0.1,
        p_edge=0.3,
    )

    print("\n===== Per-run results (head) =====")
    print(df.head())

    print("\n===== Summary by band (mean/std) =====")
    print(summary)

    print("\n===== Overall summary (mean/std) =====")
    print(summary_all)

    # save
    df.to_csv("pcmciplus_benchmark_runs.csv", index=False)
    summary.to_csv("pcmciplus_benchmark_summary_by_band.csv", index=False)
    summary_all.to_csv("pcmciplus_benchmark_summary_overall.csv")

    print("\nSaved:")
    print(" - pcmciplus_benchmark_runs.csv")
    print(" - pcmciplus_benchmark_summary_by_band.csv")
    print(" - pcmciplus_benchmark_summary_overall.csv")


if __name__ == "__main__":
    main()
