import hydra
from omegaconf import DictConfig
import datetime
import numpy as np
import pandas as pd
import pycwt as wavelet
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt

from tools.tools import load_joint_samples, benchmarking, standard_preprocessing, save_run
from tools.scoring_tools import score

# 从3.py中导入你自定义的因果发现方法
from MTSCSD.TFSCD import run_one_experiment  # 替换为3.py中的方法


# ============================================================
# 基础工具：数据处理和频谱计算
# ============================================================

def compute_spectrum(data, fs=1.0):
    N = len(data)
    freqs = np.fft.fftfreq(N, d=1 / fs)  # 频率数组
    spectrum = np.abs(np.fft.fft(data))  # 幅度谱
    half_n = N // 2
    freqs = freqs[:half_n]
    spectrum = spectrum[:half_n]
    return freqs, spectrum


def expand_frequency_range(freqs, spectrum, scale_factor=100):
    freqs_expanded = freqs * scale_factor
    spectrum_expanded = spectrum
    return freqs_expanded, spectrum_expanded


def auto_select_bands(data, fs=1.0, num_bands=3, plot=False, scale_factor=100):
    """
    自动根据数据的频谱选择频带，并放大频率范围。

    参数：
    - data: 输入数据，通常是时间序列
    - fs: 采样频率
    - num_bands: 频带数量
    - plot: 是否绘制频谱图
    - scale_factor: 放大因子

    返回：
    - freq_bands: 自动选择的频带范围
    """
    freqs, spectrum = compute_spectrum(data, fs)

    # 放大频率范围
    freqs_expanded, spectrum_expanded = expand_frequency_range(freqs, spectrum, scale_factor)

    # 计算频谱的总能量
    total_energy = np.sum(spectrum_expanded)

    # 计算频谱的累计能量（用于划分频带）
    cumulative_energy = np.cumsum(spectrum_expanded) / total_energy

    # 选择频带的边界（分割频谱的累计能量）
    band_edges = [np.searchsorted(cumulative_energy, i / num_bands) for i in range(1, num_bands)]

    # 检查并修正边界索引，确保索引不会超出频率数组的大小
    max_index = len(freqs_expanded) - 1
    band_edges = [min(edge, max_index) for edge in band_edges]

    # 生成频带
    freq_bands = {}
    prev_edge = 0
    for i, edge in enumerate(band_edges):
        band_name = f"Band {i + 1}"
        freq_bands[band_name] = (freqs_expanded[prev_edge], freqs_expanded[edge])
        prev_edge = edge

    if plot:
        # 绘制频谱图
        plt.figure(figsize=(8, 4))
        plt.plot(freqs_expanded, spectrum_expanded, label="Expanded Spectrum")
        for band in freq_bands.values():
            plt.axvline(x=band[0], color='r', linestyle='--', label=f"{band[0]} Hz")
            plt.axvline(x=band[1], color='r', linestyle='--', label=f"{band[1]} Hz")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

    return freq_bands


# ============================================================
# 计算频域因果关系
# ============================================================

def cp_fcd_bandwise_causal(data, variables, dt, mother, dj, s0, J,
                           freq_bands,
                           coh_threshold=0.2,
                           precision_threshold=None):
    if precision_threshold is None:
        precision_threshold = {k: 1 for k in freq_bands.keys()}

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

                # 将 pandas Series 转换为 numpy 数组
                data_i = np.asarray(data[var_i])
                data_j = np.asarray(data[var_j])

                # 确保数据是 C-contiguous 数组
                W1, sj, freq, coi, _, _ = wavelet.cwt(data_i, dt, dj=dj, s0=s0, J=J, wavelet=mother)
                W2, sj, freq, coi, _, _ = wavelet.cwt(data_j, dt, dj=dj, s0=s0, J=J, wavelet=mother)

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
                high2 = (WCT_b >= 0.98)

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
                    phi_adj = phi_b[high2]  # 不再修改相位，直接使用原始相位
                    w = WCT_b[high2]

                    # 使用加权平均来计算相位
                    R = np.sum(w * np.exp(1j * phi_adj)) / (np.sum(w) + 1e-12)

                    # 计算加权相位
                    weighted_phase = np.angle(R)

                    if weighted_phase < 0:  # 如果加权相位为负，则交换因果关系的方向
                        matrix[i, j] = corr_ij
                    else:  # 如果加权相位为正，则保持原方向
                        matrix[j, i] = corr_ij


        adj_df = pd.DataFrame(matrix, index=variables, columns=variables)
        corr_df = pd.DataFrame(corr_matrix, index=variables, columns=variables)

        # 偏相关（精度矩阵标准化）并删边
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
                if adj_df.loc[a, b] > 0 and strength.loc[a, b] <= precision_threshold[band_name]:
                    adj_corrected.loc[a, b] = 0.0

        corrected_matrices[band_name] = adj_corrected

    return corrected_matrices



# ============================================================
# 综合方法：结合频谱分析，自动选择频带并进行因果发现
# ============================================================

def run_causal_discovery(data, variables, dt, mother, dj, s0, J,
                         freq_bands, scale_factor=100,
                         coh_threshold=0.6, precision_threshold=None):
    freq_bands = auto_select_bands(data, fs=1 / 6, num_bands=3, scale_factor=scale_factor)
    corrected_matrices = cp_fcd_bandwise_causal(
        data, variables, dt, mother, dj, s0, J,
        freq_bands=freq_bands,
        coh_threshold=coh_threshold,
        precision_threshold=precision_threshold
    )
    print(corrected_matrices)

    # 合并三个频带的因果图矩阵为一个 0-1 矩阵
    final_causal_graph = np.zeros_like(corrected_matrices["Band 1"].values)
    for band_name in corrected_matrices:
        band_matrix = corrected_matrices[band_name].values
        final_causal_graph = np.maximum(final_causal_graph, (band_matrix > 0.5).astype(int))

    return final_causal_graph


def calculate_metrics(predicted, actual):
    """
    计算F1, TPR, SHD等指标
    """
    # Flatten the matrices to make them 1D
    predicted_flat = predicted.flatten()
    actual_flat = actual.flatten()

    # F1 score (macro average)
    f1 = f1_score(actual_flat, predicted_flat, average='macro')  # F1 score

    # Confusion matrix and True Positive Rate (TPR)
    cm = confusion_matrix(actual_flat, predicted_flat)
    tpr = cm[1, 1] / (cm[1, 0] + cm[1, 1])  # True Positive Rate (Recall)

    # Structural Hamming Distance (SHD)
    shd = np.sum(predicted_flat != actual_flat)  # SHD (simple comparison)

    return f1, tpr, shd


@hydra.main(version_base=None, config_path="config", config_name="benchmark.yaml")
def main(cfg: DictConfig):
    start = datetime.datetime.now()
    print(cfg)
    print("Loading data...")

    test_data, test_labels = load_joint_samples(
        cfg, preprocessing=standard_preprocessing if cfg.dt_preprocess else None
    )

    # 初始化存储结果的列表
    all_metrics = []

    # 遍历所有样本进行因果发现，并计算指标
    for data, labels in zip(test_data, test_labels):
        print("Performing Causal Discovery with your custom method...")
        variables = data.columns.tolist()
        dt = 1.0 / 6  # 每六小时采样一次
        mother = wavelet.Morlet(6)
        s0 = 12  # 初始尺度
        dj = 1 / 12  # 小波间隔
        J = 84  # 小波尺度数
        print(labels)


        # 调用综合方法进行因果发现
        result = run_causal_discovery(data, variables, dt, mother, dj, s0, J,
                                      freq_bands={}, scale_factor=100, coh_threshold=0.98)

        print(result)
        # 将 labels 转换为 0-1 矩阵
        labels_binary = (labels.values > 0).astype(int)

        # 计算因果图预测结果和实际标签之间的 F1, TPR, SHD 指标
        f1, tpr, shd = calculate_metrics(result, labels_binary)
        all_metrics.append((f1, tpr, shd))

    # 计算平均值和标准差
    metrics = np.array(all_metrics)
    avg_f1 = np.mean(metrics[:, 0])
    avg_tpr = np.mean(metrics[:, 1])
    avg_shd = np.mean(metrics[:, 2])

    # 计算标准差
    std_f1 = np.std(metrics[:, 0])
    std_tpr = np.std(metrics[:, 1])
    std_shd = np.std(metrics[:, 2])

    print(f"Average F1: {avg_f1} ± {std_f1}")
    print(f"Average TPR: {avg_tpr} ± {std_tpr}")
    print(f"Average SHD: {avg_shd} ± {std_shd}")

    # 保存结果
    save_run(cfg, all_metrics, start)


if __name__ == "__main__":
    main()
