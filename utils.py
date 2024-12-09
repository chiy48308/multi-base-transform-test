import librosa
import numpy as np

def extract_mel_spectrogram(audio_path, sr=16000, n_mels=80, duration=5):
    y, sr = librosa.load(audio_path, sr=sr, duration=duration)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)  # 對數幅度
    return mel_db.T  # 轉置方便模型輸入
