# varlingam_benchmark_latest.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# ===== 依赖：你的环境里需要能 import 到 VARLiNGAM =====
# 你上传的 varlingam.py 里是这样导入的：:contentReference[oaicite:1]{index=1}
from baselines.scripts_python.python_packages.lingam_master.lingam.var_lingam import VARLiNGAM


# -----------------------------
# Utils: z-score 标准化（只用 train 统计量）
# -----------------------------
def zscore_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd

def zscore_transform(X, mu, sd):
    return (X - mu) / sd

def mse_mae(y_true, y_pred):
    e = y_true - y_pred
    mse = float(np.mean(e**2))
    mae = float(np.mean(np.abs(e)))
    return mse, mae


# -----------------------------
# 合成 VAR 数据（带真值滞后系数矩阵 A_l）
# X_t = sum_{l=1..L} X_{t-l} @ A_l + noise
#
# A_l: shape [d, d], A_l[i,j] 表示 i -> j 的滞后 l 影响系数
# -----------------------------
def sample_sparse_A(d, p_edge=0.2, w_range=(0.5, 1.0), rng=None):
    rng = np.random.default_rng() if rng is None else rng
    A = np.zeros((d, d), dtype=float)
    mask = rng.random((d, d)) < p_edge
    np.fill_diagonal(mask, 0.0)
    A[mask] = rng.uniform(w_range[0], w_range[1], size=mask.sum())
    # 随机给符号
    sign = rng.choice([-1.0, 1.0], size=mask.sum())
    A[mask] *= sign
    return A

def generate_var_data(d=5, T=2000, L=5, p_edge=0.15, noise_std=0.5, burnin=200, seed=0):
    rng = np.random.default_rng(seed)

    # 生成真值 A_l
    A_list = []
    for l in range(1, L + 1):
        # 可以让不同 lag 稀疏度不同（这里统一）
        A_l = sample_sparse_A(d, p_edge=p_edge, w_range=(0.4, 1.2), rng=rng)
        A_list.append(A_l)

    # 为了稳定：缩放整体谱半径（粗略）
    # 让 sum_l ||A_l|| 不至于太大
    scale = 0.8 / max(1e-12, sum(np.linalg.norm(A, ord=2) for A in A_list))
    A_list = [A * scale for A in A_list]

    X = np.zeros((T + burnin, d), dtype=float)
    noise = rng.normal(0.0, noise_std, size=X.shape)

    for t in range(L, T + burnin):
        val = np.zeros(d, dtype=float)
        for l in range(1, L + 1):
            val += X[t - l] @ A_list[l - 1]  # [d] @ [d,d] -> [d]
        X[t] = val + noise[t]

    X = X[burnin:]  # 去 burn-in
    return X, A_list


# -----------------------------
# 将真值/估计 转成 “扩展滞后邻接矩阵”
# 形状: [d, d*L]，列按 (node, lag) 展开
# 位置 (i, j + (l-1)*d) 表示 i -> j 的 lag=l 边
# -----------------------------
def A_list_to_expanded_adj(A_list, thr=1e-12):
    L = len(A_list)
    d = A_list[0].shape[0]
    B = np.zeros((d, d * L), dtype=int)
    for l, A_l in enumerate(A_list, start=1):
        B[:, (l - 1) * d : l * d] = (np.abs(A_l) > thr).astype(int)
    return B

def varlingam_to_expanded_adj(model, d, L, thr=1e-12):
    """
    model._adjacency_matrices 是一个 list，长度=lags，每个 shape [d,d]
    这里按你 varlingam.py 的拼接方式 concat -> [d, d*L] 再二值化 :contentReference[oaicite:2]{index=2}
    """
    mats = model._adjacency_matrices  # list of [d,d]
    if mats is None or len(mats) == 0:
        return np.zeros((d, d * L), dtype=int), np.zeros((d, d * L), dtype=float)

    # 如果模型返回的 lags != L，就取 min 对齐
    L_eff = min(L, len(mats))
    am = np.concatenate([mats[i] for i in range(L_eff)], axis=1)  # [d, d*L_eff]
    if L_eff < L:
        pad = np.zeros((d, d * (L - L_eff)), dtype=float)
        am = np.concatenate([am, pad], axis=1)
    B = (np.abs(am) > thr).astype(int)
    return B, am


# -----------------------------
# 结构指标：Precision/Recall/TPR/F1 + SHD(FP+FN)
# 在“扩展滞后图”上做二分类
# -----------------------------
def structure_metrics(B_true, B_est):
    B_true = (B_true != 0).astype(int)
    B_est = (B_est != 0).astype(int)

    tp = int(np.sum((B_true == 1) & (B_est == 1)))
    fp = int(np.sum((B_true == 0) & (B_est == 1)))
    fn = int(np.sum((B_true == 1) & (B_est == 0)))

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)   # = TPR
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    shd = fp + fn

    return {
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": float(prec),
        "TPR": float(rec),
        "F1": float(f1),
        "SHD": int(shd),
        "nnz": int(np.sum(B_est))
    }


# -----------------------------
# 用 VAR 系数做滚动一步预测（重构指标）
# y_pred[t] = sum_l x[t-l] @ A_l_est
# -----------------------------
def predict_var_one_step(X, A_list_est):
    L = len(A_list_est)
    T, d = X.shape
    Y_true = X[L:]               # [T-L, d]
    Y_pred = np.zeros_like(Y_true)

    for t in range(L, T):
        val = np.zeros(d, dtype=float)
        for l in range(1, L + 1):
            val += X[t - l] @ A_list_est[l - 1]
        Y_pred[t - L] = val
    return Y_true, Y_pred


# -----------------------------
# 单次实验：生成数据 -> 训练 VARLiNGAM -> 指标
# -----------------------------
def run_one(seed=0,
            d=5, T=2000, L=5, p_edge=0.15, noise_std=0.5,
            train_ratio=0.7,
            thr_graph=0.0,
            verbose=False):

    X, A_true_list = generate_var_data(d=d, T=T, L=L, p_edge=p_edge, noise_std=noise_std, seed=seed)
    B_true = A_list_to_expanded_adj(A_true_list, thr=1e-12)

    # split
    Tn = X.shape[0]
    n_train = int(Tn * train_ratio)
    X_train = X[:n_train]
    X_test = X[n_train - L:]  # 为了一步预测需要前 L 个滞后

    # z-score（只用 train 统计量）
    mu, sd = zscore_fit(X_train)
    X_train_z = zscore_transform(X_train, mu, sd)
    X_test_z = zscore_transform(X_test, mu, sd)

    # VARLiNGAM fit（输入 shape [n, d]）
    model = VARLiNGAM(lags=L, criterion='bic', prune=True)
    model.fit(X_train_z)

    # 结构输出（扩展图）
    B_est, am = varlingam_to_expanded_adj(model, d=d, L=L, thr=thr_graph)
    sm = structure_metrics(B_true, B_est)

    # 重构：用估计的 adjacency_matrices 做一步预测
    # 注意：model._adjacency_matrices 是 list，每个 [d,d]
    A_est_list = model._adjacency_matrices
    # 对齐 L
    if A_est_list is None or len(A_est_list) == 0:
        A_est_list = [np.zeros((d, d), dtype=float) for _ in range(L)]
    else:
        if len(A_est_list) < L:
            A_est_list = list(A_est_list) + [np.zeros((d, d), dtype=float) for _ in range(L - len(A_est_list))]
        else:
            A_est_list = list(A_est_list[:L])

    Y_true, Y_pred = predict_var_one_step(X_test_z, A_est_list)
    mse, mae = mse_mae(Y_true, Y_pred)

    if verbose:
        print("seed:", seed, "F1:", sm["F1"], "SHD:", sm["SHD"], "MSE:", mse, "MAE:", mae)

    out = {**sm, "MSE": mse, "MAE": mae}
    return out


# -----------------------------
# 多轮基准 + 汇总表
# -----------------------------
def run_benchmark(n_runs=20,
                  d=5, T=2000, L=5, p_edge=0.15, noise_std=0.5,
                  train_ratio=0.7,
                  thr_graph=0.0,
                  save_csv="varlingam_benchmark_results.csv"):

    rows = []
    for r in range(n_runs):
        m = run_one(seed=r, d=d, T=T, L=L, p_edge=p_edge, noise_std=noise_std,
                    train_ratio=train_ratio, thr_graph=thr_graph, verbose=False)
        m["run"] = r
        rows.append(m)

    df = pd.DataFrame(rows)
    df.to_csv(save_csv, index=False)

    # 汇总 mean ± std
    metric_cols = ["F1", "TPR", "Precision", "SHD", "MSE", "MAE", "nnz"]
    summary = []
    for c in metric_cols:
        summary.append({
            "Metric": c,
            "Mean": float(df[c].mean()),
            "Std": float(df[c].std(ddof=1)) if len(df) > 1 else 0.0
        })
    summary_df = pd.DataFrame(summary)

    return df, summary_df


def main():
    # 你可以按需改这些超参
    df, summary_df = run_benchmark(
        n_runs=20,
        d=5, T=2000, L=5,
        p_edge=0.15,
        noise_std=0.5,
        train_ratio=0.7,
        thr_graph=0.0,  # 图阈值：0.0 表示只要非零就算边；也可设 0.1/0.2 更保守
        save_csv="varlingam_benchmark_results.csv"
    )

    print("\n==== Per-run results (head) ====")
    print(df.head())

    print("\n==== Summary (mean ± std) ====")
    print(summary_df)

    summary_df.to_csv("varlingam_benchmark_summary.csv", index=False)
    print("\nSaved:",
          "varlingam_benchmark_results.csv, varlingam_benchmark_summary.csv")


if __name__ == "__main__":
    main()
