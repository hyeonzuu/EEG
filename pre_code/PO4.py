import pandas as pd
import numpy as np
import scipy.signal as signal
import os

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

# 처리할 파일들이 있는 디렉토리 경로
input_directory = './dataset/'
output_directory = './PO4/'  # 결과 파일을 저장할 디렉토리 경로

# 디렉토리가 존재하지 않으면 생성
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 처리할 파일 목록
files_to_process = [
    'sub02_BPF_rest.csv', 'sub02_BPF_stressful.csv',
    'sub06_BPF_rest.csv', 'sub06_BPF_stressful.csv',
    'sub07_BPF_rest.csv', 'sub07_BPF_stressful.csv',
    'sub08_BPF_rest.csv', 'sub08_BPF_stressful.csv',
    'sub09_BPF_rest.csv', 'sub09_BPF_stressful.csv',
    'sub10_BPF_rest.csv', 'sub10_BPF_stressful.csv',
    'sub11_BPF_rest.csv', 'sub11_BPF_stressful.csv',
    'sub12_BPF_rest.csv', 'sub12_BPF_stressful.csv',
    'sub13_BPF_rest.csv', 'sub13_BPF_stressful.csv',
    'sub14_BPF_rest.csv', 'sub14_BPF_stressful.csv',
    'sub15_BPF_rest.csv', 'sub15_BPF_stressful.csv',
    'sub16_BPF_rest.csv', 'sub16_BPF_stressful.csv',
    'sub17_BPF_rest.csv', 'sub17_BPF_stressful.csv',
    'sub18_BPF_rest.csv', 'sub18_BPF_stressful.csv',
    'sub19_BPF_rest.csv', 'sub19_BPF_stressful.csv',
    'sub20_BPF_rest.csv', 'sub20_BPF_stressful.csv'
]

for csv_file in files_to_process:
    file_path = os.path.join(input_directory, csv_file)

    # 파일 존재 여부 확인
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}, skipping...")
        continue

    # CSV 파일에서 EEG 데이터 불러오기
    df = pd.read_csv(file_path)

    # EEG 데이터 (여기에서는 PO4 채널만 사용)
    PO4 = df.iloc[:, 5].values  # PO4 채널이 여섯 번째 열에 위치한다고 가정 (열 인덱스는 5)

    # 각 주파수 대역에 대해 필터 적용
    filtered_signals_PO4 = {}
    for band, (lowcut, highcut) in frequency_bands.items():
        filtered_signals_PO4[band] = butter_bandpass_filter(PO4, lowcut, highcut, fs, order=N)

    # 필터링된 데이터프레임 생성
    df_filtered = pd.DataFrame(filtered_signals_PO4)

    # 파일명에서 세부 정보를 추출하여 출력 파일명 생성
    base_name = os.path.splitext(csv_file)[0]
    parts = base_name.split('_')
    subject = parts[0].replace('sub', '')
    condition = parts[2]
    output_file = os.path.join(output_directory, f'sub_{subject}_{condition}_PO4_full.csv')
    
    # 필터링된 데이터를 CSV로 저장
    df_filtered.to_csv(output_file, index=False)

    print(f'Processed and saved: {output_file}')
