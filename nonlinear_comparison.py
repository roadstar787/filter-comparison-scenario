import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import multivariate_normal

# --- シミュレーション設定 ---
DT = 0.1  # タイムステップ [s]
SIM_TIME = 20.0  # シミュレーション時間 [s]
RADIUS = 5.0  # 円の半径 [m]
ANGULAR_VELOCITY = 0.5  # 角速度 [rad/s]

# --- ノイズ設定 ---
# 制御入力ノイズ
V_NOISE_STD = 0.1  # 速度の標準偏差 [m/s]
OMEGA_NOISE_STD = np.deg2rad(1.0)  # 角速度の標準偏差 [rad/s]
# 測定ノイズ
MEASUREMENT_NOISE_STD = 0.5  # 測定値の標準偏差 [m]

# --- フィルタ設定 ---
# EKF
Q_ekf = np.diag([0.1**2, 0.1**2, np.deg2rad(1.0)**2])  # プロセスノイズの共分散
R_ekf = np.diag([MEASUREMENT_NOISE_STD**2, MEASUREMENT_NOISE_STD**2])  # 測定ノイズの共分散
# PF
N_PARTICLES = 500
Q_pf = np.diag([V_NOISE_STD**2, OMEGA_NOISE_STD**2]) # PFのプロセスノイズは制御入力に対するもの

# --- モデル定義 ---
def motion_model(x, u, dt):
    """ 状態遷移モデル """
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])
    B = np.array([[dt * np.cos(x[2, 0]), 0],
                  [dt * np.sin(x[2, 0]), 0],
                  [0, dt]])
    return F @ x + B @ u

def observation_model(x):
    """ 観測モデル """
    return np.array([[x[0, 0]], [x[1, 0]]])

def jacobian_F(x, u, dt):
    """ 運動モデルのヤコビ行列 """
    v = u[0, 0]
    theta = x[2, 0]
    return np.array([
        [1.0, 0.0, -dt * v * np.sin(theta)],
        [0.0, 1.0, dt * v * np.cos(theta)],
        [0.0, 0.0, 1.0]])

def jacobian_H():
    """ 観測モデルのヤコビ行列 """
    return np.array([
        [1, 0, 0],
        [0, 1, 0]])

# --- データ生成 ---
def generate_data(sim_time, dt, radius, omega):
    time = np.arange(0, sim_time, dt)
    # 真の軌道
    true_x = radius * np.cos(omega * time)
    true_y = radius * np.sin(omega * time)
    true_theta = omega * time + np.pi / 2
    true_state = np.vstack([true_x, true_y, true_theta])
    
    # 制御入力 (真値)
    v = radius * omega
    u = np.array([[v], [omega]])
    
    # 測定値
    measurements = true_state[:2, :] + np.random.normal(0, MEASUREMENT_NOISE_STD, size=true_state[:2, :].shape)
    return time, true_state, measurements, u

# --- EKF実装 ---
def run_ekf(measurements, u):
    x_est = np.zeros((3, 1))  # 初期状態 [x, y, theta]
    P_est = np.eye(3)
    ekf_history = [x_est]

    for i in range(measurements.shape[1]):
        # 予測
        x_pred = motion_model(x_est, u, DT)
        F = jacobian_F(x_est, u, DT)
        P_pred = F @ P_est @ F.T + Q_ekf

        # 更新
        H = jacobian_H()
        z_pred = observation_model(x_pred)
        y = measurements[:, i].reshape(2, 1) - z_pred
        S = H @ P_pred @ H.T + R_ekf
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_est = x_pred + K @ y
        P_est = (np.eye(3) - K @ H) @ P_pred
        
        ekf_history.append(x_est)
    return np.array(ekf_history).squeeze().T

# --- PF実装 ---
def run_pf(measurements, u):
    # PFの初期化：最初の測定値の周りにパーティクルを分布させる
    # これにより、フィルタが現実的な初期位置から追跡を開始できる
    initial_pos = measurements[:, 0].reshape(2, 1)
    particles = np.vstack([
        initial_pos[0] + np.random.randn(N_PARTICLES) * MEASUREMENT_NOISE_STD,
        initial_pos[1] + np.random.randn(N_PARTICLES) * MEASUREMENT_NOISE_STD,
        np.random.uniform(-np.pi, np.pi, N_PARTICLES)
    ])
    weights = np.ones(N_PARTICLES) / N_PARTICLES
    
    pf_history = [np.average(particles, weights=weights, axis=1)]
    particles_history = [particles.copy()]

    for i in range(measurements.shape[1]):
        # 予測
        for j in range(N_PARTICLES):
            # 制御入力にノイズを加えてばらつきを表現
            u_noisy = u + np.sqrt(Q_pf) @ np.random.randn(2, 1)
            particles[:, j] = motion_model(particles[:, j].reshape(3, 1), u_noisy, DT).flatten()

        # 更新 (尤度計算)
        z = measurements[:, i]
        for j in range(N_PARTICLES):
            likelihood = multivariate_normal.pdf(z, mean=particles[:2, j], cov=R_ekf)
            weights[j] *= likelihood
        
        weights += 1e-300
        weights /= np.sum(weights)

        # 状態推定
        x_est = np.average(particles, weights=weights, axis=1)
        pf_history.append(x_est)
        
        # リサンプリング
        if 1.0 / np.sum(weights**2) < N_PARTICLES / 2.0:
            indexes = np.random.choice(N_PARTICLES, size=N_PARTICLES, p=weights, replace=True)
            particles = particles[:, indexes]
            weights.fill(1.0 / N_PARTICLES)
        
        particles_history.append(particles.copy())

    return np.array(pf_history).T, particles_history

# --- メイン処理 ---
if __name__ == '__main__':
    time, true_state, measurements, u = generate_data(SIM_TIME, DT, RADIUS, ANGULAR_VELOCITY)
    ekf_history = run_ekf(measurements, u)
    pf_history, particles_history = run_pf(measurements, u)

    # --- アニメーション ---
    plt.rcParams['font.family'] = 'MS Gothic'
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_title("EKFとパーティクルフィルタの比較 (円運動)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)

    # プロット要素
    ax.plot(true_state[0, :], true_state[1, :], "g-", label="真の軌道")
    ax.plot(measurements[0, :], measurements[1, :], "kx", label="測定値")
    ekf_line, = ax.plot([], [], "r-", label="EKF推定値")
    pf_line, = ax.plot([], [], "b-", label="PF推定値")
    particles_scatter = ax.scatter([], [], s=10, c='b', alpha=0.3, label="パーティクル")
    ax.legend()

    def update(i):
        ekf_line.set_data(ekf_history[0, :i+1], ekf_history[1, :i+1])
        pf_line.set_data(pf_history[0, :i+1], pf_history[1, :i+1])
        
        particles = particles_history[i]
        particles_scatter.set_offsets(particles[:2, :].T)
        
        return ekf_line, pf_line, particles_scatter

    ani = animation.FuncAnimation(fig, update, frames=len(time), blit=True, interval=50, repeat=False)
    
    print("アニメーションを 'nonlinear_animation.gif' として保存中...")
    ani.save('nonlinear_animation.gif', writer='pillow', fps=15)
    print("保存が完了しました。")
    
    plt.show()
