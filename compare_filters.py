import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 各フィルタの関数をインポート
from ekf import extended_kalman_filter
from particle_filter import particle_filter

if __name__ == '__main__':
    # --- データ生成 (共通) ---
    dt = 1.0
    true_positions = np.arange(0, 50, dt)
    # 毎回同じ乱数系列を生成するためにシードを固定
    np.random.seed(0)
    measurements = true_positions + np.random.normal(0, 0.5, len(true_positions))
    
    # --- EKFの実行 ---
    measurements_ekf = measurements.reshape(-1, 1)
    initial_state_ekf = np.array([0.0])
    initial_covariance_ekf = np.array([[1.0]])
    process_noise_ekf = np.array([[1e-2]])
    measurement_noise_ekf = np.array([[0.25]])
    F_ekf = np.array([[1.0]])
    H_ekf = np.array([[1.0]])

    ekf_states, _ = extended_kalman_filter(
        measurements_ekf, initial_state_ekf, initial_covariance_ekf, 
        process_noise_ekf, measurement_noise_ekf, F_ekf, H_ekf
    )
    # EKFの推定値は初期値を含むので、プロット用に調整
    ekf_estimated_positions = ekf_states[1:, 0]

    # --- PFの実行 ---
    n_particles = 1000
    process_noise_std_pf = np.sqrt(0.1)
    measurement_noise_std_pf = np.sqrt(0.25)
    initial_state_range_pf = (0, 1)

    pf_estimated_positions = particle_filter(
        measurements, n_particles, process_noise_std_pf, 
        measurement_noise_std_pf, initial_state_range_pf
    )

    # --- アニメーションの作成 ---
    plt.rcParams['font.family'] = 'MS Gothic'
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_xlim(0, len(true_positions))
    ax.set_ylim(np.min(measurements) - 1, np.max(measurements) + 1)
    ax.set_xlabel('時間ステップ')
    ax.set_ylabel('位置')
    ax.set_title('EKFとパーティクルフィルタの比較アニメーション')
    ax.grid(True)

    # プロット要素の初期化
    true_line, = ax.plot([], [], 'g-', linewidth=2, label='真の位置')
    measurement_points, = ax.plot([], [], 'ko', markersize=3, label='測定値')
    ekf_line, = ax.plot([], [], 'r-', linewidth=2, label='推定値 (EKF)')
    pf_line, = ax.plot([], [], 'b--', linewidth=2, label='推定値 (PF)')
    ax.legend()

    def update(frame):
        # データをフレームごとに更新
        time_steps = np.arange(frame + 1)
        true_line.set_data(time_steps, true_positions[:frame + 1])
        measurement_points.set_data(time_steps, measurements[:frame + 1])
        ekf_line.set_data(time_steps, ekf_estimated_positions[:frame + 1])
        pf_line.set_data(time_steps, pf_estimated_positions[:frame + 1])
        return true_line, measurement_points, ekf_line, pf_line

    # アニメーションの生成
    ani = animation.FuncAnimation(fig, update, frames=len(true_positions), 
                                  blit=True, interval=50, repeat=False)

    # アニメーションをGIFとして保存
    print("アニメーションを 'animation.gif' として保存中...")
    ani.save('animation.gif', writer='pillow', fps=15)
    print("保存が完了しました。")

    plt.show()
