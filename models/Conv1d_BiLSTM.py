import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d_BiLSTM(nn.Module):
    def __init__(self, len_subseq, n_class=2):
        super(Conv1d_BiLSTM, self).__init__()
        
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
            nn.Dropout(0.2)
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

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(len_subseq * hidden_size // 16, (len_subseq - 1) * n_class)


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

        fc  = self.fc(self.flatten(bilstm)).view(-1, self.n_class)
        
        return F.softmax(fc, dim=-1)
