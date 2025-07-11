import numpy as np

class KalmanFilter:
    def __init__(self, F, H, Q, R, x0, P0):
        """
        カルマンフィルタの初期化

        :param F: 状態遷移行列
        :param H: 観測行列
        :param Q: プロセスノイズの共分散行列
        :param R: 観測ノイズの共分散行列
        :param x0: 初期状態推定値
        :param P0: 初期誤差共分散行列
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self):
        """
        状態の予測
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        """
        観測値による状態の更新

        :param z: 観測値
        """
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P
        return self.x

if __name__ == '__main__':
    # 1次元の運動モデルでカルマンフィルタをテストする
    
    # 時間ステップ
    dt = 1.0

    # 状態遷移行列 (x_k = F * x_{k-1})
    # 状態ベクトルは [位置, 速度]
    F = np.array([[1, dt],
                  [0, 1]])

    # 観測行列 (z_k = H * x_k)
    # 位置のみを観測
    H = np.array([[1, 0]])

    # プロセスノイズの共分散行列
    # 加速度のばらつきを仮定
    q = 0.1
    Q = np.array([[(dt**4)/4, (dt**3)/2],
                  [(dt**3)/2, dt**2]]) * q**2

    # 観測ノイズの共分散行列
    R = np.array([[0.1]])

    # 初期状態
    x0 = np.array([[0], [0]])
    P0 = np.eye(2)

    # カルマンフィルタのインスタンスを作成
    kf = KalmanFilter(F, H, Q, R, x0, P0)

    # シミュレーションデータを作成
    num_steps = 50
    true_x = np.zeros((num_steps, 2, 1))
    measurements = np.zeros((num_steps, 1, 1))
    
    # 真の状態を生成
    for i in range(1, num_steps):
        true_x[i] = F @ true_x[i-1] + np.sqrt(Q) @ np.random.randn(2, 1)

    # 観測値を生成
    for i in range(num_steps):
        measurements[i] = H @ true_x[i] + np.sqrt(R) @ np.random.randn(1, 1)

    # カルマンフィルタを実行
    estimates = []
    for z in measurements:
        kf.predict()
        estimate = kf.update(z)
        estimates.append(estimate.flatten())
    
    estimates = np.array(estimates)

    # 結果をプロット
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(range(num_steps), true_x[:, 0], label='True Position')
    plt.plot(range(num_steps), measurements[:, 0], 'o', label='Measurements')
    plt.plot(range(num_steps), estimates[:, 0], label='Kalman Filter Estimate')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.title('Kalman Filter Simulation')
    plt.legend()
    plt.grid(True)
    plt.show()
