import torch
import torch.nn as nn


class CNNRNNClassifier(nn.Module):
    def __init__(
        self, num_classes=2, input_freq_bins=128, hidden_size=128, rnn_layers=1
    ):
        super(CNNRNNClassifier, self).__init__()

        # CNN Block (simple 2-layer CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 1, F, T) -> (B, 32, F, T)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (B, 32, F/2, T/2)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (B, 64, F/4, T/4)
        )

        self.freq_out = input_freq_bins // 4  # after 2x (2x2) max-pooling on frequency
        self.rnn_input_dim = (
            64 * self.freq_out
        )  # each time step becomes a flattened conv feature

        self.rnn = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),  # bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: [B, F, T] → [B, 1, F, T]
        x = x.unsqueeze(1)

        # CNN
        x = self.cnn(x)  # [B, C, F', T']

        # prepare for RNN
        x = x.permute(0, 3, 1, 2)  # [B, T', C, F']
        x = x.flatten(2)  # [B, T', C*F']

        # RNN
        rnn_out, _ = self.rnn(x)  # [B, T', 2*hidden_size]
        final_output = rnn_out[:, -1, :]  # Take last time step

        # Classifier
        return self.classifier(final_output)
