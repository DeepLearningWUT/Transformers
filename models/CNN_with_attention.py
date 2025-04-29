import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, rnn_outputs):
        # rnn_outputs: [batch_size, time_steps, hidden_dim]
        scores = self.attention(rnn_outputs)  # [batch_size, time_steps, 1]
        scores = scores.squeeze(-1)  # [batch_size, time_steps]
        weights = torch.softmax(scores, dim=1)  # [batch_size, time_steps]

        context = torch.bmm(weights.unsqueeze(1), rnn_outputs)  # [batch_size, 1, hidden_dim]
        context = context.squeeze(1)  # [batch_size, hidden_dim]
        return context

class CNNRNNWithAttention(nn.Module):
    def __init__(
        self, num_classes=3, input_freq_bins=128, hidden_size=64, rnn_layers=1
    ):
        super(CNNRNNWithAttention, self).__init__()

        # CNN Block (light version)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.2),
        )

        self.freq_out = input_freq_bins // 4
        self.rnn_input_dim = 32 * self.freq_out

        self.rnn = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(0.3 if rnn_layers > 1 else 0.0),
        )

        self.attention = AttentionLayer(hidden_size * 2)

        self.dropout_rnn_out = nn.Dropout(p=0.3)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, F, T] -> [B, 1, F, T]
        x = self.cnn(x)     # [B, C, F', T']

        x = x.permute(0, 3, 1, 2)  # [B, T', C, F']
        x = x.flatten(2)           # [B, T', C*F']

        rnn_out, _ = self.rnn(x)  # [B, T', 2*hidden_size]

        context = self.attention(rnn_out)  # [B, 2*hidden_size]

        context = self.dropout_rnn_out(context)

        return self.classifier(context)
