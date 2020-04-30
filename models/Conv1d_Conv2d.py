import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d_Conv2d(nn.Module):
    def __init__(self, len_subseq, n_class=2):
        super(Conv1d_Conv2d, self).__init__()
        
        self.n_class = n_class
        
        seq = (len_subseq, 20)
        pssm = (21, len_subseq)
                
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

            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Conv1d(128, 128, 3, padding=1)
        )
        
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.4),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.4),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Flatten(),
            nn.Linear(32 * ((((128 - 2) // 2) - 2) // 2 - 2) * ((((len_subseq - 2) // 2) - 2) // 2 - 2), 64),
            nn.Linear(64, (len_subseq - 1) * n_class)
        )
        
        
    def forward(self, seq, pssm):
        batch_size = seq.shape[0]
        
        seq = torch.transpose(seq, dim0=1, dim1=2)
        
        features = torch.cat([seq, pssm], dim=-2)

        conv1d = self.conv1d(features).unsqueeze(1)

        conv2d = self.conv2d(conv1d).view(-1, self.n_class)
        
        return F.softmax(conv2d, dim=-1)
