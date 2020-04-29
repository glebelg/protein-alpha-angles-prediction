import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM(nn.Module):
    def __init__(self, len_subseq, n_class=2):
        super(BiLSTM, self).__init__()
        
        self.n_class = n_class

        seq = (len_subseq, 20)
        pssm = (21, len_subseq)
        
        hidden_size = 512

        self.bilstm1 = nn.LSTM(input_size=seq[1] + pssm[0],
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
        
        pssm = torch.transpose(pssm, dim0=1, dim1=2)

        features = torch.cat([seq, pssm], dim=-1)

        bilstm, _ = self.bilstm1(features)
        bilstm, _ = self.bilstm2(bilstm)
        bilstm, _ = self.bilstm3(bilstm)
        bilstm, _ = self.bilstm4(bilstm)
        bilstm, _ = self.bilstm5(bilstm)

        fc = self.fc(self.flatten(bilstm)).view(-1, self.n_class)
        
        return F.softmax(fc, dim=-1)
