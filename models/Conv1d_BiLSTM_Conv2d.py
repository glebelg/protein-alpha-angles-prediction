import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d_BiLSTM_Conv2d(nn.Module):
    def __init__(self, len_subseq, n_class=2):
        super(Conv1d_BiLSTM_Conv2d, self).__init__()
        
        self.n_class = n_class

        seq = (len_subseq, 20)
        pssm = (21, len_subseq)
        
        hidden_size = 512
        
        self.conv1d = nn.Sequential(
            nn.Conv1d(seq[1] + pssm[0], 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),

            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),            

            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
        )

        self.bilstm1 = nn.LSTM(input_size=64,
                               hidden_size=hidden_size // 2,
                               bidirectional=True,
                               batch_first=True,
                               dropout=0.4)
            
        self.bilstm2 = nn.LSTM(input_size=hidden_size,
                               hidden_size=hidden_size // 4,
                               bidirectional=True,
                               batch_first=True,
                               dropout=0.4)
            
        self.bilstm3 = nn.LSTM(input_size=hidden_size // 2,
                               hidden_size=hidden_size // 8,
                               bidirectional=True,
                               batch_first=True,
                               dropout=0.4)
            
        self.bilstm4 = nn.LSTM(input_size=hidden_size // 4,
                               hidden_size=hidden_size // 16,
                               bidirectional=True,
                               batch_first=True,
                               dropout=0.2)
        
        self.bilstm5 = nn.LSTM(input_size=hidden_size // 8,
                               hidden_size=hidden_size // 32,
                               bidirectional=True,
                               batch_first=True,
                               dropout=0.2)

        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.Conv2d(8, 8, 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.2),
            
            nn.Conv2d(8, 16, 3),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.2),

            nn.Conv2d(16, 32, 3),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * (len_subseq - 12) * (hidden_size // 16 - 12), 64)
        self.fc2 = nn.Linear(64, (len_subseq - 1) * n_class)


    def forward(self, seq, pssm):
        batch_size = seq.shape[0]
        
        seq = torch.transpose(seq, dim0=1, dim1=2)
        
        features = torch.cat([seq, pssm], dim=-2)

        conv1d = self.conv1d(features)
        conv1d = torch.transpose(conv1d, dim0=1, dim1=2)
        
        bilstm, _ = self.bilstm1(conv1d)
        bilstm, _ = self.bilstm2(bilstm)
        bilstm, _ = self.bilstm3(bilstm)
        bilstm, _ = self.bilstm4(bilstm)
        bilstm, _ = self.bilstm5(bilstm)

        conv2d = self.conv2d(bilstm.unsqueeze(1))

        fc1 = self.fc1(self.flatten(conv2d))
        fc2 = self.fc2(fc1).view(-1, self.n_class)

        return F.softmax(fc2, dim=-1)
