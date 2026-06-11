import numpy as np
import pandas as pd
import pycwt as wavelet
import matplotlib
import torch
from sklearn.metrics import mean_squared_error

from MTSCSD.lbfgsb_scipy import LBFGSBScipy
from MTSCSD.trace_expm import trace_expm


# ============================================================
# 基础工具
# ============================================================

def set_random_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def safe_array(x):
    """把 NaN/Inf 清掉，避免 matmul / metric 出现 invalid value warning。"""
    x = np.asarray(x)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def evaluate_signal(x_true, x_pred):
    x_true = np.real(x_true)
    x_pred = np.real(x_pred)
    x_true = (x_true - x_true.mean()) / x_true.std()
    x_pred = (x_pred - x_pred.mean()) / x_pred.std()
    mse = mean_squared_error(x_true, x_pred)
    mae = np.mean(np.abs(x_true - x_pred))

    return mse, mae

def count_accuracy_no_dag_check(B_true, B_est):
    """
    基于 NOTEARS 的 count_accuracy 改写：不强制 B_est 必须 DAG（避免阈值后轻微成环就报错）。
    返回：fdr, tpr, fpr, shd, nnz
    """
    B_true = (B_true != 0).astype(int)
    B_est = (B_est != 0).astype(int)
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

    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}


# ============================================================
# 合成数据（沿用你原逻辑）
# ============================================================

def sample_random_dag(d=3, p_edge=0.3, w_min=0.7 , w_max=0.9, rng=None):
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
                       bandwidth_ratio=0.15, chirp_ratio=0.1,
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
def generate_freqs_dict(d, band_ranges, rng=None, integer=True, margin=0.5):
    """
    根据变量个数 d 和频段范围，自动生成互不相同的中心频率

    Parameters
    ----------
    d : int
        变量个数
    band_ranges : dict
        e.g. {"Low": (0.1, 6), "Mid": (7, 15), "High": (17, 25)}
    rng : np.random.Generator
        随机数生成器（用于可复现）
    integer : bool
        是否生成整数频率（True 更稳）
    margin : float
        远离频段边界的安全余量（Hz）

    Returns
    -------
    freqs_dict : dict[str, list[float]]
    """
    rng = np.random.default_rng() if rng is None else rng
    freqs_dict = {}

    for band, (fmin, fmax) in band_ranges.items():
        lo = fmin + margin
        hi = fmax - margin

        if integer:
            candidates = np.arange(np.ceil(lo), np.floor(hi) + 1)
            if len(candidates) < d:
                raise ValueError(
                    f"[{band}] 频段内可用整数频率不足 {d} 个，请扩大频段或减小 d"
                )
            freqs = rng.choice(candidates, size=d, replace=False)
        else:
            # 连续频率版本（一般不推荐）
            freqs = rng.uniform(lo, hi, size=d)

        freqs_dict[band] = list(np.sort(freqs))

    return freqs_dict

def generate_data_simple(duration=20, fs=500, d=5,
                         band_ranges=None, noise_std=0.1, rng=None,
                         base_mode="narrowband_noise",
                         phi_shared_map=None, show=False):
    if band_ranges is None:
        band_ranges = {"Low": (0.1, 6), "Mid": (7, 15), "High": (17, 25)}
    rng = np.random.default_rng() if rng is None else rng
    T = int(duration * fs)
    dt = 1.0 / fs
    t = np.arange(T) / fs

    freqs_dict = generate_freqs_dict(
        d=d,
        band_ranges=band_ranges,
        rng=rng,  # 和整体数据生成共用 RNG，保证可复现
        integer=False,  # 强烈建议
        margin=0.5
    )
    if phi_shared_map is None:
        phi_shared_map = {k: 0.0 for k in band_ranges}

    A_bands, tau_bands, comps = {}, {}, {}
    X_total = np.zeros((d, T))

    for band in band_ranges.keys():
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
        for i in range(d):
            for j in range(d):
                if i != j and A[i, j] != 0:
                    X_band[j] += A[i, j] * roll_shift(Z[i], tau[i, j], dt=dt, mode="zero")

        comps[band] = X_band
        X_total += X_band

        if show:
            print(f"\n==== {band} band ====")
            print("A:\n", A)
            print("tau:\n", np.round(tau, 2))

    X_total += rng.normal(0, noise_std, size=X_total.shape)
    data = {f"x{i+1}": X_total[i] for i in range(d)}
    return data, t, X_total, A_bands, tau_bands, comps, dt


# ============================================================
# 频域因果结构：WCT + 相位 + 偏相关删边（保留你原逻辑）
# ============================================================

def cp_fcd_bandwise_causal(data, variables, dt, mother, dj, s0, J,
                           freq_bands,
                           coh_threshold=0.6,
                           precision_threshold=None):
    if precision_threshold is None:
        precision_threshold = {k: 0.4 for k in freq_bands.keys()}

    d = len(variables)
    corrected_matrices = {}

    for band_name, (fmin, fmax) in freq_bands.items():
        matrix = np.zeros((d, d))
        corr_matrix = np.zeros((d, d))

        for i, var_i in enumerate(variables):
            for j, var_j in enumerate(variables):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    continue

                W1, sj, freq, coi, _, _ = wavelet.cwt(data[var_i], dt, dj=dj, s0=s0, J=J, wavelet=mother)
                W2, sj, freq, coi, _, _ = wavelet.cwt(data[var_j], dt, dj=dj, s0=s0, J=J, wavelet=mother)

                scales = sj[:, None]
                S1 = mother.smooth((np.abs(W1) ** 2) / scales, dt, dj, sj)
                S2 = mother.smooth((np.abs(W2) ** 2) / scales, dt, dj, sj)
                W12 = W1 * np.conj(W2)
                S12 = mother.smooth(W12 / scales, dt, dj, sj)

                fmask = (freq >= fmin) & (freq <= fmax)
                if not np.any(fmask):
                    corr_matrix[i, j] = 0.0
                    continue

                S1_b, S2_b, S12_b = S1[fmask, :], S2[fmask, :], S12[fmask, :]
                phi_b = np.angle(W12[fmask, :])

                WCT_b = (np.abs(S12_b) ** 2) / (S1_b * S2_b + 1e-12)

                high = (WCT_b >= coh_threshold)
                high2 = (WCT_b >= 0.8)

                weights = WCT_b * (WCT_b >= coh_threshold)
                wsum = np.sum(weights)
                if wsum <= 1e-12:
                    corr_ij = 0.0
                else:
                    cov12 = np.sum(weights * np.real(S12_b)) / wsum
                    var1 = np.sum(weights * S1_b) / wsum
                    var2 = np.sum(weights * S2_b) / wsum
                    corr_ij = cov12 / (np.sqrt(var1 * var2) + 1e-12)
                    corr_ij = np.sqrt(np.abs(corr_ij))
                    corr_ij = float(np.clip(corr_ij, 0, 1.0))
                corr_matrix[i, j] = corr_ij

                if np.sum(high) > 0 and np.sum(high2) > 0:
                    phi_adj = np.where(phi_b[high2] < 0, phi_b[high2] + np.pi, phi_b[high2])
                    w = WCT_b[high2]
                    R = np.sum(w * np.exp(1j * phi_adj)) / (np.sum(w) + 1e-12)
                    weighted_phase = np.angle(R)

                    if 0 < weighted_phase < np.pi / 2:
                        matrix[i, j] = corr_ij
                    elif -np.pi / 2 < weighted_phase < 0:
                        matrix[j, i] = corr_ij
                    elif np.pi / 2 < weighted_phase < np.pi:
                        matrix[j, i] = corr_ij
                    elif -np.pi < weighted_phase < -np.pi / 2:
                        matrix[i, j] = corr_ij

        adj_df = pd.DataFrame(matrix, index=variables, columns=variables)
        corr_df = pd.DataFrame(corr_matrix, index=variables, columns=variables)

        # 偏相关（精度矩阵标准化）
        cov_like = corr_df.values
        try:
            prec = np.linalg.inv(cov_like)
        except np.linalg.LinAlgError:
            prec = np.linalg.pinv(cov_like)

        eps = 1e-12
        ddiag = np.sqrt(np.clip(np.diag(prec), eps, None))
        scale = np.outer(ddiag, ddiag)
        partial = -prec / np.clip(scale, eps, None)
        np.fill_diagonal(partial, 1.0)
        partial_df = pd.DataFrame(partial, index=variables, columns=variables)

        # 删边
        adj_corrected = adj_df.copy()
        strength = partial_df.abs()
        for a in adj_df.index:
            for b in adj_df.columns:
                if a == b:
                    continue
                if adj_df.loc[a, b] > 0 and strength.loc[a, b] < precision_threshold[band_name]:
                    adj_corrected.loc[a, b] = 0.0

        corrected_matrices[band_name] = adj_corrected

    return corrected_matrices


# ============================================================
# CWT / ICWT：按频段 mask 拼接完整 W_pred 后一次 ICWT（你强调的正确逻辑）
# ============================================================

def compute_cwt_all(data, variables, dt, dj, s0, J, mother):
    cwt_results = {}
    for var in variables:
        W, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(data[var], dt, dj, s0, J, mother)
        cwt_results[var] = {"W": W, "scales": scales, "freqs": freqs, "coi": coi}
    return cwt_results

def reconstruct_from_bands(
    A_bands_est,
    cwt_results,
    variables,
    freq_bands,
    dt,
    dj,
    mother,
    self_weight=1.0
):
    """
    正确的小波频段重构方式：
    - 每个变量 j：
        * 从原始 W_j 开始
        * 在每个频段内：用 (自身 + 父节点线性组合) 替换
        * 最后对完整 W_pred_j 做一次 icwt
    """
    d = len(variables)
    freqs = cwt_results[variables[0]]["freqs"]
    scales = cwt_results[variables[0]]["scales"]
    T = cwt_results[variables[0]]["W"].shape[1]

    X_pred = np.zeros((d, T))

    for j, var_j in enumerate(variables):
        # 1️⃣ 从原始小波系数开始（关键）
        W_pred_j = cwt_results[var_j]["W"].copy()

        for band_name, (fmin, fmax) in freq_bands.items():
            fmask = (freqs >= fmin) & (freqs <= fmax)
            if not np.any(fmask):
                continue

            A_k = A_bands_est[band_name]  # [d,d]

            # 2️⃣ 先保留自身项（非常重要）
            W_band = self_weight * cwt_results[var_j]["W"][fmask, :].copy()

            # 3️⃣ 加父节点贡献
            for i, var_i in enumerate(variables):
                if i == j:
                    continue
                if abs(A_k[i, j]) < 1e-8:
                    continue
                W_band += A_k[i, j] * cwt_results[var_i]["W"][fmask, :]

            # 4️⃣ 替换该频段
            W_pred_j[fmask, :] = W_band

        # 5️⃣ 逆小波变换（此时应为实数）
        x_pred_j = wavelet.icwt(W_pred_j, scales, dt, dj, mother)

        # 数值保险（理论上虚部应≈0）
        X_pred[j] = np.real(x_pred_j)

    return X_pred

# ============================================================
# NOTEARS（LBFGSB + 增广拉格朗日）：同时优化三个频段矩阵
# - 加一个轻微的“贴近初值”项，避免 L1 把所有边压成 0
# ============================================================

def optimize_three_band_notears_lbfgsb(corrected_matrices, freq_bands,
                                      l1_lambda=1e-3,
                                      fit_lambda=1e-1,
                                      neg_lambda=10.0,
                                      diag_lambda=10.0,
                                      max_outer_iter=20,
                                      h_tol=1e-8,
                                      rho_max=1e16):
    band_names = list(freq_bands.keys())
    d = corrected_matrices[band_names[0]].shape[0]

    # 初值
    A0 = {bn: torch.tensor(corrected_matrices[bn].values, dtype=torch.double) for bn in band_names}

    # 待优化参数（用 clone+detach 避免你遇到的 torch.tensor(tensor) warning）
    A = {bn: A0[bn].clone().detach().requires_grad_(True) for bn in band_names}

    optimizer = LBFGSBScipy([A[bn] for bn in band_names])

    rho, alpha, h_prev = 1.0, 0.0, np.inf

    def h_func():
        # 三个频段的无环性约束之和（你想“体现所有矩阵”）
        h = 0.0
        for bn in band_names:
            h = h + (trace_expm(A[bn] * A[bn]) - d)
        return h

    for _ in range(max_outer_iter):
        def closure():
            optimizer.zero_grad()

            # 增广拉格朗日项
            h_val = h_func()
            lag = alpha * h_val + 0.5 * rho * h_val * h_val

            # L1 稀疏
            l1 = 0.0
            # 贴近初值（避免塌到全0）
            fit = 0.0
            # 对角为0、非负（用软惩罚，不改变参数化）
            diag_pen = 0.0
            neg_pen = 0.0

            for bn in band_names:
                Abn = A[bn]
                l1 = l1 + l1_lambda * torch.norm(Abn, p=1)
                fit = fit + 0.5 * fit_lambda * torch.sum((Abn - A0[bn]) ** 2)
                diag_pen = diag_pen + diag_lambda * torch.sum(torch.diag(Abn) ** 2)
                neg_pen = neg_pen + neg_lambda * torch.sum(torch.relu(-Abn))

            total = lag + l1 + fit + diag_pen + neg_pen
            total.backward()
            return total

        optimizer.step(closure)  # 注意：不要依赖 step 的返回值（可能是 None）

        with torch.no_grad():
            h_new = float(h_func().item())
        if h_new > 0.25 * h_prev:
            rho *= 10.0
        alpha += rho * h_new
        if h_new <= h_tol or rho >= rho_max:
            break
        h_prev = h_new

    A_est = {}
    for bn in band_names:
        mat = A[bn].detach().cpu().numpy()
        mat = safe_array(mat)
        np.fill_diagonal(mat, 0.0)
        mat[np.abs(mat) < 1e-3] = 0.0
        A_est[bn] = mat

    return A_est


# ============================================================
# 单次实验：合成 → 频域初值 → NOTEARS(LBFGSB) → 重构 → 指标
# ============================================================

def run_one_experiment(seed=0,
                       duration=20, fs=500, d=5,
                       band_ranges=None,
                       l1_lambda=1e-3,
                       fit_lambda=1e-1,
                       edge_threshold=0.4):
    set_random_seed(seed)
    if band_ranges is None:
        band_ranges = {"Low": (0.1, 6), "Mid": (7, 15), "High": (17, 25)}

    # 合成数据
    data, t, X_total, A_true_bands, tau_bands, comps, dt = generate_data_simple(
        duration=duration, fs=fs, d=d,
        band_ranges=band_ranges, noise_std=0.1,
        base_mode="narrowband_noise",
        show=False
    )
    variables = [f"x{i+1}" for i in range(d)]

    # 小波参数
    mother = wavelet.Morlet(6)
    s0 = 2 * dt
    dj = 1 / 12
    J = int(7 / dj)

    # 频域初值邻接（WCT+phase+偏相关删边）
    corrected = cp_fcd_bandwise_causal(
        data, variables, dt, mother, dj, s0, J,
        freq_bands=band_ranges,
        coh_threshold=0.6,
        precision_threshold={k: 0.4 for k in band_ranges.keys()}
    )

    # CWT
    cwt_results = compute_cwt_all(data, variables, dt, dj, s0, J, mother)

    # NOTEARS：三个频段联合优化
    A_est = optimize_three_band_notears_lbfgsb(
        corrected, band_ranges,
        l1_lambda=l1_lambda,
        fit_lambda=fit_lambda,
        max_outer_iter=20,
        h_tol=1e-8
    )

    # 重构：拼接频段系数后一次 ICWT（整段时间序列）
    X_pred = reconstruct_from_bands(A_est, cwt_results, variables, band_ranges, dt, dj, mother)

    # 重构指标（对变量取平均）
    mse_list, mae_list = [], []
    for i in range(d):
        mse_i, mae_i = evaluate_signal(X_total[i], X_pred[i])
        mse_list.append(mse_i)
        mae_list.append(mae_i)
    mse_mean = float(np.mean(mse_list))
    mae_mean = float(np.mean(mae_list))

    # 结构指标：每频段计算，然后平均
    F1_list, TPR_list, SHD_list = [], [], []
    for bn in band_ranges.keys():
        B_true = (A_true_bands[bn] > 0).astype(int)
        B_est = (np.abs(A_est[bn]) > edge_threshold).astype(int)

        acc = count_accuracy_no_dag_check(B_true, B_est)
        precision = 1.0 - acc["fdr"]
        recall = acc["tpr"]
        F1 = 0.0 if (precision + recall) <= 0 else 2 * precision * recall / (precision + recall)

        F1_list.append(float(F1))
        TPR_list.append(float(acc["tpr"]))
        SHD_list.append(float(acc["shd"]))

    return {
        "F1": float(np.mean(F1_list)),
        "TPR": float(np.mean(TPR_list)),
        "SHD": float(np.mean(SHD_list)),
        "MSE": mse_mean,
        "MAE": mae_mean,
    }


# ============================================================
# 多轮实验 + 表格
# ============================================================

def main():
    matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    N_RUNS = 30  # 你可以改成 50/100
    rows = []
    for r in range(N_RUNS):
        metrics = run_one_experiment(
            seed=1234 + r,
            duration=20,
            fs=500,
            d=3,
            l1_lambda=1e-3,      # 太大很容易全 0
            fit_lambda=1e-1,     # 让优化别塌缩（贴近初值）
            edge_threshold=0.4
        )
        rows.append(metrics)
        print(f"Run {r}: {metrics}")

    df = pd.DataFrame(rows)
    df.loc["mean"] = df.mean(numeric_only=True)
    df.loc["std"] = df.std(numeric_only=True)
    print("\n===== Summary =====")
    print(df)

    # df.to_csv("cp_fcd_results.csv", index=True, float_format="%.6f")

if __name__ == "__main__":
    main()
