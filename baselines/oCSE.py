import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_squared_error

# ============================================================
# 1) oCSE (与你文件风格一致的实现：基于 Causation Entropy)
#    说明：它得到的是 lag-1 的因果： x_i(t-1) -> x_j(t)
# ============================================================

def _as_2d_Tk(a):
    """
    Ensure array is shaped as [T, k].
    Accepts:
      - 1D [T]  -> [T,1]
      - 2D [T,k] stays
      - pandas Series/DataFrame: works via np.asarray
    """
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    elif a.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape={a.shape}")
    return a

def entropy(x):
    """
    Differential entropy for (assumed) Gaussian variable X:
      H(X) = 0.5 * log( (2πe)^k * det(Cov(X)) )
    x: array-like [T,k] or [T]
    """
    x = _as_2d_Tk(x)  # [T,k]
    k = x.shape[1]

    # covariance over variables (columns)
    cov = np.cov(x, rowvar=False, bias=False)  # -> (k,k) if x is 2D
    cov = np.atleast_2d(cov)                   # ensure (k,k) even when k=1

    # numerical jitter
    cov = cov + 1e-12 * np.eye(k)

    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        # fallback: increase jitter a bit (still minimal intrusion)
        cov = cov + 1e-8 * np.eye(k)
        sign, logdet = np.linalg.slogdet(cov)

    return 0.5 * (k * np.log(2 * np.pi * np.e) + logdet)


def conditional_entropy(y, x):
    """
    H(Y|X) = H([Y,X]) - H(X)   under joint Gaussian assumption.
    y: [T, ky] or [T]
    x: [T, kx] or [T]
    """
    y = _as_2d_Tk(y)
    x = _as_2d_Tk(x)
    if y.shape[0] != x.shape[0]:
        raise ValueError(f"Length mismatch: y has {y.shape[0]} rows, x has {x.shape[0]} rows")

    yx = np.hstack([y, x])  # [T, ky+kx]
    return entropy(yx) - entropy(x)

def causation_entropy(q, p, c=None):
    """
    CSE(q <- p | c) for lag-1:
    q(t) compared with p(t-1) conditioned on c(t-1)
    q, p: pandas Series or 1D array, length T
    c: pandas DataFrame or 2D array, shape [T, k] or None
    """
    q = np.asarray(q).astype(float)
    p = np.asarray(p).astype(float)

    qt = q[1:]      # q(t)
    pt_1 = p[:-1]   # p(t-1)

    if c is None:
        # CSE = H(q|empty) - H(q|p)
        # 这里用差分形式： CSE = H(q|c) - H(q|[c,p])
        # c为空时等价于 H(q) - H(q|p)
        return entropy(qt) - conditional_entropy(qt, pt_1)

    c = np.asarray(c).astype(float)
    ct_1 = c[:-1, :]  # c(t-1)

    # CSE(q <- p | c) = H(q|c) - H(q|c,p)
    return conditional_entropy(qt, ct_1) - conditional_entropy(qt, np.hstack([ct_1, pt_1.reshape(-1, 1)]))

def ocse(data_df, sig_level=0.05, verbose=False):
    """
    oCSE 主程序：对每个目标变量 q，找其父集合 Pa(q)。
    返回 parents_df: index=target, column 'parents' 是 list
    """
    variables = list(data_df.columns)
    T = data_df.shape[0]

    parents = {q: [] for q in variables}

    # Stage 1: Aggregative discovery (greedy add)
    for q in variables:
        # candidate set excluding q
        candidates = [p for p in variables if p != q]
        S = []  # selected parents

        while True:
            best_p = None
            best_cse = -np.inf

            for p in candidates:
                if p in S:
                    continue
                if len(S) == 0:
                    cse_val = causation_entropy(data_df[q], data_df[p], c=None)
                else:
                    cse_val = causation_entropy(data_df[q], data_df[p], c=data_df[S].values)
                if cse_val > best_cse:
                    best_cse = cse_val
                    best_p = p

            # significance test by Gaussian approx:
            # test statistic ~ sqrt(T)*CSE, assume normal under H0 (rough heuristic)
            # 你也可以换成 permutation test；这里保持轻量、可多轮跑
            z = np.sqrt(max(T - 1, 1)) * best_cse
            pval = 1.0 - norm.cdf(z)

            if verbose:
                print(f"[Stage1] target={q}, best={best_p}, cse={best_cse:.4g}, z={z:.3g}, p={pval:.3g}")

            if pval < sig_level:
                S.append(best_p)
            else:
                break

        parents[q] = S

    # Stage 2: Progressive removal (prune)
    for q in variables:
        S = parents[q].copy()
        if len(S) <= 1:
            continue
        # try remove each parent p if CSE becomes insignificant when conditioning on remaining
        changed = True
        while changed:
            changed = False
            for p in S.copy():
                C = [x for x in S if x != p]
                if len(C) == 0:
                    cse_val = causation_entropy(data_df[q], data_df[p], c=None)
                else:
                    cse_val = causation_entropy(data_df[q], data_df[p], c=data_df[C].values)

                z = np.sqrt(max(T - 1, 1)) * cse_val
                pval = 1.0 - norm.cdf(z)
                if verbose:
                    print(f"[Stage2] target={q}, test_remove={p}, cse={cse_val:.4g}, p={pval:.3g}")
                if pval >= sig_level:
                    S.remove(p)
                    changed = True

        parents[q] = S

    parents_df = pd.DataFrame({"parents": [parents[q] for q in variables]}, index=variables)
    return parents_df

def parents_df_to_adj(parents_df, variables):
    """
    parents_df: index=target, col 'parents' list of sources
    输出 B_est: [d,d] binary adj, B[i,j]=1 means i -> j
    """
    d = len(variables)
    idx = {v: k for k, v in enumerate(variables)}
    B = np.zeros((d, d), dtype=int)
    for tgt in variables:
        j = idx[tgt]
        for src in parents_df.loc[tgt, "parents"]:
            i = idx[src]
            if i != j:
                B[i, j] = 1
    return B

# ============================================================
# 2) 合成数据：3 个频段各自 DAG + (滞后混合) + 频段叠加
# ============================================================

def sample_random_dag(d=3, p_edge=0.3, w_min=0.7, w_max=0.9, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    while True:
        order = rng.permutation(d)
        W = np.zeros((d, d), dtype=float)
        for u_rank in range(d):
            for v_rank in range(u_rank + 1, d):
                u, v = order[u_rank], order[v_rank]
                if rng.random() < p_edge:
                    W[u, v] = rng.uniform(w_min, w_max)
        if np.any(W != 0):
            return W

def roll_shift_int(x, lag):
    """整数滞后：前 lag 位置补 0，避免环绕"""
    y = np.roll(x, lag)
    if lag > 0:
        y[:lag] = 0.0
    return y

def generate_multiband_data(
    duration=20.0,
    fs=200,
    d=4,
    noise_std=0.2,
    band_names=("Low", "Mid", "High"),
    freqs_dict=None,
    rng=None,
):
    """
    生成 X(t)=sum_k X^k(t)+noise
    每个频段 k：
      base: z_i^k(t)=sin(2pi f_i^k t + phi_i^k) + small noise
      mix : for each edge i->j in W^k, add w_ij^k * z_i^k(t - tau_ij^k)
    返回：
      data_df: [T,d]
      W_true_bands: dict band -> weighted adj [d,d]
      B_true_bands: dict band -> binary adj [d,d]
    """
    rng = np.random.default_rng() if rng is None else rng
    T = int(duration * fs)
    t = np.arange(T) / fs

    if freqs_dict is None:
        # 你也可以换成你论文里那套频率设置
        freqs_dict = {
            "Low":  np.linspace(1.0, 3.0, d),
            "Mid":  np.linspace(6.0, 10.0, d),
            "High": np.linspace(14.0, 18.0, d),
        }

    X_total = np.zeros((d, T), dtype=float)
    W_true_bands = {}
    B_true_bands = {}

    for band in band_names:
        Wk = sample_random_dag(d=d, rng=rng)
        Bk = (Wk != 0).astype(int)
        W_true_bands[band] = Wk
        B_true_bands[band] = Bk

        freqs = np.array(freqs_dict[band], dtype=float)
        phi = rng.uniform(0, 2*np.pi, size=d)

        Z = np.zeros((d, T), dtype=float)
        for i in range(d):
            Z[i] = np.sin(2*np.pi*freqs[i]*t + phi[i]) + 0.05 * rng.normal(size=T)

        Xk = Z.copy()
        for i in range(d):
            for j in range(d):
                if i == j or Wk[i, j] == 0:
                    continue
                # 简单设置：滞后随频率变化（高频滞后更小），保证是整数
                max_lag = max(1, int(fs / max(freqs[i], 1e-6) / 6))
                lag = int(rng.integers(1, max_lag + 1))
                Xk[j] += Wk[i, j] * roll_shift_int(Z[i], lag)

        X_total += Xk

    X_total += noise_std * rng.normal(size=X_total.shape)

    variables = [f"x{i+1}" for i in range(d)]
    data_df = pd.DataFrame({variables[i]: X_total[i] for i in range(d)})
    return data_df, W_true_bands, B_true_bands

# ============================================================
# 3) 结构指标：F1 / TPR / SHD
# ============================================================

def metrics_f1_tpr_shd(B_true, B_est):
    """
    B_true, B_est: [d,d] binary, i->j
    SHD = extra + missing + reversed (与常见定义一致)
    """
    B_true = (B_true != 0).astype(int)
    B_est = (B_est != 0).astype(int)
    d = B_true.shape[0]

    # TP/FP/FN
    tp = int(np.sum((B_true == 1) & (B_est == 1)))
    fp = int(np.sum((B_true == 0) & (B_est == 1)))
    fn = int(np.sum((B_true == 1) & (B_est == 0)))

    f1 = (2 * tp) / max((2 * tp + fp + fn), 1)
    tpr = tp / max((tp + fn), 1)

    # reversed: predicted i->j but true has j->i (and not i->j)
    rev = 0
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if B_est[i, j] == 1 and B_true[i, j] == 0 and B_true[j, i] == 1:
                rev += 1

    # skeleton extra/missing
    pred_skel = (B_est + B_est.T) > 0
    true_skel = (B_true + B_true.T) > 0

    extra = int(np.sum(pred_skel & (~true_skel)) // 2)
    missing = int(np.sum((~pred_skel) & true_skel) // 2)

    shd = extra + missing + rev
    return {"F1": f1, "TPR": tpr, "SHD": shd}

# ============================================================
# 4) 重构指标：用 lag-1 线性预测（符合 oCSE 的边定义）
#    x_j(t) ~ sum_{i in Pa(j)} beta_ij * x_i(t-1)
# ============================================================

def reconstruction_mse_mae(data_df, parents_df):
    variables = list(data_df.columns)
    data_df = (data_df - data_df.mean()) / data_df.std()
    T = data_df.shape[0]

    X = data_df.values  # [T,d]
    d = X.shape[1]
    # predict for t=1..T-1
    y_true_all = []
    y_pred_all = []

    for j, var_j in enumerate(variables):
        pa = parents_df.loc[var_j, "parents"]
        y = X[1:, j]  # [T-1]
        if len(pa) == 0:
            # no parent: predict 0 (or mean); 这里用 mean 更稳定
            yhat = np.full_like(y, fill_value=np.mean(y))
        else:
            cols = [variables.index(p) for p in pa]
            Z = X[:-1, cols]  # [T-1, |pa|]
            # 最小二乘
            Z_aug = np.hstack([Z, np.ones((Z.shape[0], 1))])  # add bias
            beta, *_ = np.linalg.lstsq(Z_aug, y, rcond=None)
            yhat = Z_aug @ beta

        y_true_all.append(y)
        y_pred_all.append(yhat)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    mse = mean_squared_error(y_true_all, y_pred_all)
    mae = float(np.mean(np.abs(y_true_all - y_pred_all)))
    return {"MSE": float(mse), "MAE": mae}

# ============================================================
# 5) 多轮实验：只跑 oCSE，输出 5 个指标，并保存表格
# ============================================================

def run_ocse_benchmark(
    n_runs=20,
    d=4,
    duration=20.0,
    fs=200,
    sig_level=0.05,
    noise_std=0.2,
    seed0=123,
    out_csv="ocse_metrics.csv",
    verbose=False
):
    rows = []
    for r in range(n_runs):
        rng = np.random.default_rng(seed0 + r)

        data_df, W_true_bands, B_true_bands = generate_multiband_data(
            duration=duration, fs=fs, d=d, noise_std=noise_std, rng=rng
        )
        variables = list(data_df.columns)

        parents_df = ocse(data_df, sig_level=sig_level, verbose=False)
        B_est = parents_df_to_adj(parents_df, variables)

        # 结构指标：按你的要求，把 B_est 当成 Low/Mid/High 都一样去算
        for band_name, B_true in B_true_bands.items():
            m_struct = metrics_f1_tpr_shd(B_true, B_est)
            m_rec = reconstruction_mse_mae(data_df, parents_df)  # 单尺度重构（全时域）
            rows.append({
                "run": r,
                "band": band_name,
                **m_struct,
                **m_rec
            })

        if verbose:
            print(f"[run {r}] done. edges={int(B_est.sum())}")

    df = pd.DataFrame(rows)

    # 你最终要的“5个指标”：F1 / TPR / SHD / MSE / MAE
    metrics = ["F1", "TPR", "SHD", "MSE", "MAE"]
    summary = df.groupby("band")[metrics].agg(["mean", "std"])
    summary_all = df[metrics].agg(["mean", "std"])

    df.to_csv(out_csv, index=False)

    return df, summary, summary_all

def main():
    df, summary, summary_all = run_ocse_benchmark(
        n_runs=30,
        d=3,
        duration=20.0,
        fs=200,
        sig_level=0.05,
        noise_std=0.2,
        seed0=123,
        out_csv="ocse_metrics.csv",
        verbose=True
    )

    print("\n===== Per-band mean±std =====")
    print(summary)

    print("\n===== Overall mean±std (all bands pooled) =====")
    print(summary_all)

    print("\nSaved:", "ocse_metrics.csv")
    print("\nHead of per-run table:")
    print(df.head())

if __name__ == "__main__":
    main()
