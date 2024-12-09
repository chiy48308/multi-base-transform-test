import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from dataset import LibriSpeechDataset, collate_fn

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 更新數據集以限制音頻長度並過濾無效數據
class LibriSpeechDatasetOptimized(LibriSpeechDataset):
    def __init__(self, data_dir, sr=16000, n_mels=80, duration=5):
        super().__init__(data_dir, sr, n_mels, duration)
        self.data = [(path, text) for path, text in self.data if os.path.exists(path)]
        print(f"Filtered dataset size: {len(self.data)}")

    def __getitem__(self, idx):
        audio_path, text = self.data[idx]
        waveform, _ = torchaudio.load(audio_path)
        max_length = 16000 * 5  # 限制為 5 秒音頻
        if waveform.size(1) > max_length:
            waveform = waveform[:, :max_length]
        return waveform.squeeze(0), text, waveform.shape[-1]

# 自定義 collate_fn，增加空檢查
def collate_fn(batch):
    if len(batch) == 0:
        raise ValueError("Empty batch encountered in collate_fn.")

    mels = [item[0].unsqueeze(1) if item[0].dim() == 1 else item[0] for item in batch]  # 確保每個 `mel` 是 2D
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


# 音頻編碼器
class AudioEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(AudioEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU()
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: [batch_size, 1, time_steps, features]
        x = self.cnn(x)  # Convolution: [batch_size, hidden_dim, reduced_time_steps, reduced_features]
        x = x.flatten(2).permute(0, 2, 1)  # Reshape to [batch_size, seq_length, hidden_dim]
        x = self.transformer(x)  # Transformer Encoder
        return x

# 文本編碼器
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_heads, num_layers):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_dim, hidden_dim)

    def forward(self, x):
        x = self.embedding(x)  # Embedding: [batch_size, seq_length, embed_dim]
        x = self.fc(x)  # Linear transformation
        x = self.transformer(x)  # Transformer Encoder
        return x

# 多編碼器結構
class MultiEncoderModel(nn.Module):
    def __init__(self, audio_input_dim, text_vocab_size, embed_dim, hidden_dim, num_heads, num_layers):
        super(MultiEncoderModel, self).__init__()
        self.audio_encoder = AudioEncoder(audio_input_dim, hidden_dim, num_heads, num_layers)
        self.text_encoder = TextEncoder(text_vocab_size, embed_dim, hidden_dim, num_heads, num_layers)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, audio, text):
        audio_features = self.audio_encoder(audio)
        text_features = self.text_encoder(text)

        min_seq_len = min(audio_features.size(1), text_features.size(1))
        audio_features = audio_features[:, :min_seq_len, :]
        text_features = text_features[:, :min_seq_len, :]

        attn_output, _ = self.attention(text_features, audio_features, audio_features)
        sentence_representation = attn_output.mean(dim=1)
        score = torch.sigmoid(self.fc(sentence_representation))
        return score

# 訓練邏輯
def train_model(loader, model, optimizer, loss_fn, total_epochs=5):
    print("Starting  model training...")
    use_mixed_precision = torch.cuda.is_available()
    scaler = GradScaler() if use_mixed_precision else None
    epoch_losses = []

    for epoch in range(total_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (waveform, text, _) in enumerate(loader):
            waveforms = waveform.to(torch.float32).to(device)
            text = torch.randint(0, 1000, (waveform.size(0), 10)).to(device)  # Simulate text input
            labels = torch.rand(waveform.size(0), 1).to(device)

            optimizer.zero_grad()
            if use_mixed_precision:
                with autocast():
                    outputs = model(waveforms, text)
                    loss = loss_fn(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(waveforms, text)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}/{total_epochs}, Batch {batch_idx + 1}/{len(loader)}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")
    return epoch_losses

# 繪製損失曲線
def plot_loss_curve(epoch_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Training Loss Curve", fontsize=16)
    plt.grid()
    plt.show()

# 主程序
if __name__ == "__main__":
    print("Initializing dataset...")
    dataset = LibriSpeechDatasetOptimized("LibriSpeech")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)  # Reduce batch size to 1
    print("Dataset initialized.")

    print("Initializing model...")
    model = MultiEncoderModel(
        audio_input_dim=80, text_vocab_size=1000, embed_dim=256,
        hidden_dim=512, num_heads=8, num_layers=3
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    print("Starting training...")
    epoch_losses = train_model(loader, model, optimizer, loss_fn)
    plot_loss_curve(epoch_losses)
