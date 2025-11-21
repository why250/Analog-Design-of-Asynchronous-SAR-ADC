import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- 1. 参数设置 ---
Fs = 1000            # 采样频率 (Hz)
T = 1.0 / Fs         # 采样周期 (s)
L = 1500             # 信号长度 (采样点数)
t = np.arange(0, L) * T  # 时间向量 (s)

# --- 2. 生成信号 ---
# 创建一个复合信号，包含一个50Hz和一个120Hz的正弦波
S = 0.7 * np.sin(2 * np.pi * 50 * t) + 1.0 * np.sin(2 * np.pi * 120 * t)
# 添加高斯白噪声
noise = 1.5 * np.random.randn(len(t))
signal_with_noise = S + noise

# --- 3. 执行快速傅里叶变换 (FFT) ---
Y = np.fft.fft(signal_with_noise)

# --- 4. 处理 FFT 结果 ---
P2 = np.abs(Y / L)
P1 = P2[0:L//2+1] # 
P1[1:-1] = 2 * P1[1:-1]
f = Fs * np.arange(0, L//2 + 1) / L # np.arrange(0,3) return array([0,1,2])

# --- 5. 自动寻找峰值 ---
# 使用 scipy.signal.find_peaks 函数
# prominence 参数非常有用，它代表一个峰值比其周围的波谷高出多少。
# 这可以有效过滤掉噪声中的小尖峰。您可以根据噪声水平调整这个值。
peaks, _ = find_peaks(P1, prominence=0.5) # 关键步骤：prominence值越大，筛选条件越苛刻

# --- 6. 打印结果 ---
print("找到的主要频率峰值:")
print("-" * 30)
if len(peaks) > 0:
    for i in peaks:
        freq = f[i]
        amplitude = P1[i]
        print(f"频率: {freq:.2f} Hz, 振幅: {amplitude:.2f}")
else:
    print("未找到显著的频率峰值。")
print("-" * 30)


# --- 7. 绘图 ---
plt.figure(figsize=(12, 8))

# 绘制时域信号
plt.subplot(2, 1, 1)
plt.plot(t, signal_with_noise)
plt.title('Time Domain Signal (with Noise)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# 绘制频域频谱并标记峰值
plt.subplot(2, 1, 2)
plt.plot(f, P1)
# 在找到的峰值位置绘制红色 'x' 标记
plt.plot(f[peaks], P1[peaks], "x", color='red', markersize=10, label='Detected Peaks')
plt.title('Single-Sided Amplitude Spectrum with Detected Peaks')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude |P1(f)|')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()