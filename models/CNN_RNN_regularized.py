import torch
import torch.nn as nn


class CNNRNNClassifierDropout(nn.Module):
    def __init__(
        self, num_classes=2, input_freq_bins=128, hidden_size=128, rnn_layers=1
    ):
        super(CNNRNNClassifierDropout, self).__init__()

        # CNN Block (simple 2-layer CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=0.2),  # ➔ ajout dropout après premier MaxPool
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=0.2),  # ➔ ajout dropout après deuxième MaxPool
        )

        self.freq_out = input_freq_bins // 4  # after 2x (2x2) max-pooling on frequency
        self.rnn_input_dim = 64 * self.freq_out

        self.rnn = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(
                0.3 if rnn_layers > 1 else 0.0
            ),  # ➔ GRU interne dropout si plusieurs couches
        )

        self.dropout_rnn_out = nn.Dropout(p=0.3)  # ➔ Dropout entre RNN et Classifier

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, F, T] -> [B, 1, F, T]
        x = self.cnn(x)  # [B, C, F', T']

        x = x.permute(0, 3, 1, 2)  # [B, T', C, F']
        x = x.flatten(2)  # [B, T', C*F']

        rnn_out, _ = self.rnn(x)  # [B, T', 2*hidden_size]
        final_output = rnn_out[:, -1, :]  # Last time step output

        final_output = self.dropout_rnn_out(final_output)  # ➔ petit dropout ici

        return self.classifier(final_output)
