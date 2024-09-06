import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# CSV 파일에서 EEG 데이터 불러오기
csv_file = './dataset/sub10_BPF_rest.csv'  # CSV 파일 경로를 지정하세요
df = pd.read_csv(csv_file)


T7 = df.iloc[:, 2].values

# 샘플링 주파수
fs = 500

# 필터의 차수
N = 4

# 주파수 대역 정의
frequency_bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 30),
    'Gamma': (30, 70)
}

# Butterworth 대역 통과 필터 설계 함수
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

# 필터 적용 함수
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y



# 각 주파수 대역에 대해 필터 적용
filtered_signals_T7 = {}
for band, (lowcut, highcut) in frequency_bands.items():
    filtered_signals_T7[band] = butter_bandpass_filter(T7, lowcut, highcut, fs, order=N)

# 필터링된 데이터 저장 (모든 대역을 한 번에 CSV 파일로 저장)
df_filtered = pd.DataFrame(filtered_signals_T7)
df_filtered.to_csv('sub_10_rest_AF4_full.csv', index=False)

# 시간 벡터 생성
t = np.arange(len(T7)) / fs

# # 결과 시각화
# plt.figure(figsize=(12, 18))
#
# plt.subplot(6, 1, 1)
# plt.plot(t, T7, label='Original T7')
# plt.title('[09 rest] Original T7')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.legend()
#
# plot_idx = 2
# for band in frequency_bands.keys():
#     plt.subplot(6, 1, plot_idx)
#     plt.plot(t, filtered_signals_AF4[band],
#              label=f'{band} Band AF4 ({frequency_bands[band][0]}-{frequency_bands[band][1]} Hz)', color='orange')
#     plt.title(f'{band} Amplitude AF4')
#     plt.xlabel('Time [s]')
#     plt.ylabel('Amplitude')
#     plt.legend()
#     plot_idx += 1
#
# plt.tight_layout()
# plt.show()
