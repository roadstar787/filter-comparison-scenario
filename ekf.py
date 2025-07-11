import numpy as np
import matplotlib.pyplot as plt

def extended_kalman_filter(measurements, initial_state, initial_covariance, process_noise, measurement_noise, F, H):
    """
    Extended Kalman Filter (EKF) の実装。
    この例では線形システムを扱うため、標準的なカルマンフィルタと同じになります。

    Args:
        measurements (numpy.ndarray): 測定値のリスト。形状は (n_samples, dim_z)。
        initial_state (numpy.ndarray): 初期状態推定 (x_0)。
        initial_covariance (numpy.ndarray): 初期共分散行列 (P_0)。
        process_noise (numpy.ndarray): プロセスノイズの共分散行列 (Q)。
        measurement_noise (numpy.ndarray): 測定ノイズの共分散行列 (R)。
        F (numpy.ndarray): 状態遷移行列。
        H (numpy.ndarray): 観測行列。

    Returns:
        tuple:  推定された状態のリストと共分散のリスト。
    """
    x = initial_state
    P = initial_covariance

    states = [x]
    covariances = [P]

    for z in measurements:
        # --- 予測ステップ ---
        # 状態の予測
        x_pred = F @ x
        # 共分散の予測
        P_pred = F @ P @ F.T + process_noise

        # --- 更新ステップ ---
        # イノベーション（観測残差）
        y = z - H @ x_pred
        # イノベーションの共分散
        S = H @ P_pred @ H.T + measurement_noise
        # カルマンゲイン
        K = P_pred @ H.T @ np.linalg.inv(S)

        # 状態の更新
        x = x_pred + K @ y
        # 共分散の更新
        P = (np.eye(len(x)) - K @ H) @ P_pred

        states.append(x)
        covariances.append(P)

    return np.array(states), np.array(covariances)


if __name__ == '__main__':
    # 1次元の状態で位置を推定する簡単な例
    # 状態: [位置]
    # 測定値: [位置]

    dt = 1.0
    # 測定値の生成 (例: ノイズが乗った等速直線運動)
    true_positions = np.arange(0, 100, dt)
    measurements = true_positions + np.random.normal(0, 0.5, len(true_positions))
    measurements = measurements.reshape(-1, 1)  # 形状を (n_samples, 1) にする

    # EKFのパラメータ設定
    # 初期状態ベクトル [位置]
    initial_state = np.array([0.0])
    # 初期共分散行列
    initial_covariance = np.array([[1.0]])
    # プロセスノイズの共分散行列 Q (モデルの不確かさ)
    process_noise = np.array([[1e-2]])
    # 測定ノイズの共分散行列 R (センサーの不確かさ)
    measurement_noise = np.array([[0.25]]) # (0.5)^2
    # 状態遷移行列 F (x_k = 1 * x_{k-1})
    F = np.array([[1.0]])
    # 観測行列 H (z_k = 1 * x_k)
    H = np.array([[1.0]])

    estimated_states, estimated_covariances = extended_kalman_filter(
        measurements, initial_state, initial_covariance, process_noise, measurement_noise, F, H
    )

    print("測定値\t\t推定された状態 (位置)")
    print("-" * 40)
    # 最初の状態は初期値なので、ループは1から始める
    for i in range(len(measurements)):
        print(f"{measurements[i][0]:.4f}\t\t{estimated_states[i+1][0]:.4f}")

    # 最終的な推定共分散
    print("\n最終的な推定共分散:")
    print(estimated_covariances[-1])

    # 結果をプロット
    # 日本語フォントの設定
    plt.rcParams['font.family'] = 'MS Gothic'
    plt.figure(figsize=(10, 6))
    plt.plot(true_positions, 'g-', label='真の位置')
    plt.plot(measurements, 'bo', markersize=4, label='測定値')
    plt.plot(estimated_states[1:, 0], 'r-', label='推定された位置 (EKF)')
    plt.title('拡張カルマンフィルタによる位置推定')
    plt.xlabel('時間ステップ')
    plt.ylabel('位置')
    plt.legend()
    plt.grid(True)
    plt.show()
