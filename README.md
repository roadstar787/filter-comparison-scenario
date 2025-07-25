# 非線形状態推定：拡張カルマンフィルターとパーティクルフィルターの比較シナリオ

このシナリオでは、円運動を行うオブジェクトの状態を推定するために、拡張カルマンフィルター（EKF）とパーティクルフィルター（PF）を実装し、その性能をアニメーションで比較します。

## 目的

- 非線形な運動モデルに対する状態推定問題を扱います。
- EKFとPFという代表的な非線形フィルタリング手法を実装し、その違いを理解します。
- フィルターの挙動（とくにパーティクルフィルターの確率分布の表現）を視覚的に確認します。
- フィルターの性能における、適切な初期化の重要性を学びます。

## ファイル構成

このシナリオは、以下のスクリプトで構成されています。

- `ekf.py`: 1次元の線形カルマンフィルターの基本的な実装です。（初期ステップ）
- `particle_filter.py`: 1次元のパーティクルフィルターの基本的な実装です。（初期ステップ）
- `compare_filters.py`: 1次元の線形モデルにおいて、EKFとPFの結果を静的に比較します。（中間ステップ）
- `nonlinear_comparison.py`: 2次元の非線形な円運動モデルを対象に、EKFとPFを実装し、比較アニメーションを生成します。（NumPyによるベクトル化版）
- `optimized_comparison.py`: **（推奨メインファイル）** `nonlinear_comparison.py` のパーティクルフィルター実装を **Numba** でコンパイルし、計算を大幅に高速化したバージョンです。
- `animation.gif`: `compare_filters.py` によって生成された、1次元線形モデルのアニメーション。
- `nonlinear_animation.gif`: `nonlinear_comparison.py` または `optimized_comparison.py` によって生成された、最終的な円運動モデルのアニメーション。

## 実行方法

1.  **必要なライブラリをインストールします。**
    ターミナルで以下のコマンドを実行してください。
    ```bash
    pip install numpy matplotlib scipy Pillow numba
    ```

2.  **メインスクリプトを実行します。**
    以下のコマンドで、Numbaによって高速化されたバージョンの比較アニメーションを生成します。
    ```bash
    python optimized_comparison.py
    ```

3.  **結果を確認します。**
    スクリプトを実行すると、プロジェクトのルートディレクトリに `nonlinear_animation.gif` というファイルが生成されます。このGIFファイルを開くと、フィルターの動作を確認できます。

## コードの要点解説

### 非線形モデル (`nonlinear_comparison.py`)

- **運動モデル**: オブジェクトの状態を `[x, y, 方位]` の3次元で定義し、一定の速度と角速度で円運動するモデルを実装しています。このモデルは三角関数を含むため非線形です。
- **観測モデル**: 状態のうち、位置 `(x, y)` のみを観測できると仮定しています。

### 拡張カルマンフィルター (EKF)

- EKFは、非線形な運動モデルを**ヤコビ行列**を用いて線形近似することで、カルマンフィルターの枠組みを適用します。
- `jacobian_F` 関数で運動モデルの、`jacobian_H` 関数で観測モデルのヤコビ行列を計算しています。
- 線形化誤差が蓄積すると、推定精度が劣化する可能性があります。

### パーティクルフィルター (PF)

- PFは、**多数のパーティクル（粒子）**を用いて確率分布を近似する手法です。非線形・非ガウス性の問題にも対応できる強力な手法です。
- **初期化**: `run_pf` 関数の冒頭では、**最初の測定値**の周りにパーティクルを初期配置しています。これは、フィルターが現実的な状態から追跡を開始するために非常に重要です。不適切な初期化は、フィルターの発散を引き起こします。
- **予測**: 各パーティクルを、ノイズを加えた制御入力に基づいて個別に動かします。
- **更新**: 測定値と各パーティクルの位置との「近さ（尤度）」を計算し、各パーティクルの**重み**を更新します。
- **リサンプリング**: 重みが小さい（見込みのない）パーティクルを淘汰し、重みが大きい（有望な）パーティクルを複製することで、効率的に確率分布を表現します。

### 高速化 (`optimized_comparison.py`)

- `optimized_comparison.py` では、計算負荷の大きいパーティクルフィルターの関数に **Numba** の `@jit(nopython=True)` デコレータを付与しています。
- これにより、該当部分のPythonコードがJust-In-Time (JIT) コンパイルされ、C言語に匹敵する速度で実行されます。
- NumPyのベクトル化だけでは到達できないレベルの高速化を実現しています。

このシナリオを通じて、状態推定フィルタリングの奥深さと面白さを体験できます。
