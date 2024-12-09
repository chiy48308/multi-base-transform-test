import os
import torch
from torch.utils.data import Dataset
import torchaudio


class LibriSpeechDataset(Dataset):
    def __init__(self, data_dir, sr=16000, n_mels=80, duration=5):
        print(f"Initializing LibriSpeechDataset with data_dir={data_dir}, sr={sr}, n_mels={n_mels}, duration={duration}")
        self.data = []
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration

        # 逐層搜尋資料夾內的 trans.txt
        transcript_dict = {}
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".trans.txt"):  # 尋找 transcript 文件
                    trans_file = os.path.join(root, file)
                    # print(f"Found transcript file: {trans_file}")

                    # 讀取 transcript 文件內容
                    with open(trans_file, "r") as f:
                        for line in f:  # 不限制行數
                            parts = line.strip().split(" ", 1)
                            if len(parts) == 2:
                                idx, text = parts
                                transcript_dict[idx] = text

        print(f"Loaded {len(transcript_dict)} entries from all transcript files.")

        # 遍歷資料夾內的全部音檔並匹配文本
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".flac"):  # 只處理 .flac 文件
                    audio_path = os.path.join(root, file)
                    audio_idx = file.replace(".flac", "")  # 提取音檔索引
                    if audio_idx in transcript_dict:  # 匹配 transcript 的文本
                        self.data.append((audio_path, transcript_dict[audio_idx]))
        print(f"Total matched data: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data) or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        audio_path, text = self.data[idx]
        mel_features = self.extract_mel_spectrogram(audio_path)
        return mel_features, text

    def extract_mel_spectrogram(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_mels=self.n_mels
        )(waveform)
        mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        return mel_spectrogram_db.squeeze(0).T  # (時間步數, 頻帶數)


def collate_fn(batch):
    mels = [item[0] for item in batch]
    texts = [item[1] for item in batch]

    # 找到最大時間步數
    max_time_steps = max(mel.shape[0] for mel in mels)
    n_mels = mels[0].shape[1]

    # 補零處理
    mels_padded = torch.zeros(len(batch), max_time_steps, n_mels)
    lengths = []
    for i, mel in enumerate(mels):
        lengths.append(mel.shape[0])
        mels_padded[i, :mel.shape[0], :] = mel

    return mels_padded, texts, lengths


# 測試使用的根目錄
data_dir = "/Users/chris/_論文研究專案2/LibriSpeech_EN_Model-Training/LibriSpeech-Transformer-Pytorch/LibriSpeech/train-clean-100"
dataset = LibriSpeechDataset(data_dir)
print(f"Dataset size: {len(dataset)}")
