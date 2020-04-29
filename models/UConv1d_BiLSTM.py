import torch
import torch.nn as nn
import torch.nn.functional as F


class UConv1d_BiLSTM(nn.Module):
    def __init__(self, len_subseq, n_class=2):
        super(UConv1d_BiLSTM, self).__init__()

        self.n_class = n_class
        
        seq = (len_subseq, 20)
        pssm = (21, len_subseq)

        hidden_size = 512
        
        self.pool = nn.MaxPool1d(2)
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(seq[1] + pssm[0], 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2)
        )

        self.deconv1 = nn.Sequential(
            nn.Upsample(size=len_subseq // 2),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        
        self.deconv2 = nn.Sequential(
            nn.Upsample(size=len_subseq),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2)
        )
        
        self.deconv3 = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2)
        )

        self.bilstm1 = nn.LSTM(input_size=32,
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
        self.fc = nn.Linear(2 * len_subseq * hidden_size // 16, (len_subseq - 1) * n_class)
        

    def forward(self, seq, pssm):
        batch_size = seq.shape[0]
        
        seq = torch.transpose(seq, dim0=1, dim1=2)
        
        features = torch.cat([seq, pssm], dim=-2)

        conv1 = self.conv1(features)

        conv2 = self.conv2(conv1)
        pool2 = self.pool(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool(conv3)

        conv4 = self.conv4(pool3)

        deconv1 = self.deconv1(conv4)
        cat1 = torch.cat([conv3, deconv1], dim=-1)

        deconv2 = self.deconv2(cat1)
        cat2 = torch.cat([conv2, deconv2], dim=-1)
        
        deconv3 = self.deconv3(cat2)
        deconv3 = torch.transpose(deconv3, dim0=1, dim1=2)

        bilstm, _ = self.bilstm1(deconv3)
        bilstm, _ = self.bilstm2(bilstm)
        bilstm, _ = self.bilstm3(bilstm)
        bilstm, _ = self.bilstm4(bilstm)
        bilstm, _ = self.bilstm5(bilstm)

        fc  = self.fc(self.flatten(bilstm)).view(-1, self.n_class)
        
        return F.softmax(fc, dim=-1)
