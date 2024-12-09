import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim):
        super(SimpleTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)  # 分數輸出（0-1）

    def forward(self, x):
        x = self.transformer(x)  # [批量, 時間步, 特徵]
        x = self.fc(x.mean(dim=1))  # 時間維度取均值
        return torch.sigmoid(x)  # 評分範圍 (0, 1)
