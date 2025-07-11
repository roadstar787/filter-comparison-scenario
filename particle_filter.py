import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def particle_filter(measurements, n_particles, process_noise_std, measurement_noise_std, initial_state_range):
    """
    Particle Filter (PF) の実装。

    Args:
        measurements (numpy.ndarray): 測定値のリスト。
        n_particles (int): パーティクルの数。
        process_noise_std (float): プロセスノイズの標準偏差。
        measurement_noise_std (float): 測定ノイズの標準偏差。
        initial_state_range (tuple): 初期状態の範囲 (min, max)。

    Returns:
        numpy.ndarray: 推定された状態のリスト。
    """
    # 1. 初期化
    # 指定された範囲でパーティクルを均一に初期化
    particles = np.random.uniform(initial_state_range[0], initial_state_range[1], n_particles)
    weights = np.ones(n_particles) / n_particles

    states = []

    for z in measurements:
        # 2. 予測
        # 各パーティクルをプロセスモデルに従って移動させ、ノイズを加える
        # x_t = F * x_{t-1} + w_{t-1}, F=1
        particles += np.random.normal(0, process_noise_std, n_particles)

        # 3. 更新
        # 尤度を計算し、重みを更新
        # 尤度は測定値と各パーティクルの予測値との間の確率密度関数で計算
        likelihoods = norm.pdf(z, loc=particles, scale=measurement_noise_std)
        weights *= likelihoods
        weights += 1.e-300  # 重みがゼロになるのを防ぐ
        weights /= np.sum(weights)  # 重みを正規化

        # 4. 状態推定
        # 重み付き平均で現在の状態を推定
        mean_state = np.sum(particles * weights)
        states.append(mean_state)

        # 5. リサンプリング
        # 重みに基づいてパーティクルをリサンプリング (系統的リサンプリング)
        N = n_particles
        positions = (np.arange(N) + np.random.rand()) / N
        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        particles = particles[indexes]
        weights.fill(1.0 / N)

    return np.array(states)


if __name__ == '__main__':
    # 1次元の状態で位置を推定する簡単な例
    dt = 1.0
    true_positions = np.arange(0, 50, dt)
    measurements = true_positions + np.random.normal(0, 0.5, len(true_positions))

    # PFのパラメータ設定
    n_particles = 1000
    process_noise_std = np.sqrt(0.1)  # EKFのQに対応
    measurement_noise_std = np.sqrt(0.25) # EKFのRに対応
    initial_state_range = (0, 1)

    estimated_states = particle_filter(
        measurements, n_particles, process_noise_std, measurement_noise_std, initial_state_range
    )

    # 結果をプロット
    plt.rcParams['font.family'] = 'MS Gothic'
    plt.figure(figsize=(10, 6))
    plt.plot(true_positions, 'g-', label='真の位置')
    plt.plot(measurements, 'bo', markersize=4, label='測定値')
    plt.plot(estimated_states, 'r-', label='推定された位置 (PF)')
    plt.title('パーティクルフィルタによる位置推定')
    plt.xlabel('時間ステップ')
    plt.ylabel('位置')
    plt.legend()
    plt.grid(True)
    plt.show()
