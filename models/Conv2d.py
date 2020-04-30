import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(self, len_subseq, n_class=2):
        super(Conv2d, self).__init__()
        
        self.n_class = n_class
        
        seq = (len_subseq, 20)
        pssm = (21, len_subseq)
                
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Flatten(),
            nn.Linear(128 * (((len_subseq - 2) // 2 - 2) // 2 - 4) * ((((seq[1] + pssm[0]) - 2) // 2 - 2) // 2 - 4), 64),
            nn.Linear(64, (len_subseq - 1) * n_class)
        )
         
        
    def forward(self, seq, pssm):
        batch_size = seq.shape[0]
        
        pssm = torch.transpose(pssm, dim0=1, dim1=2)
        
        features = torch.cat([seq, pssm], dim=-1).unsqueeze(1)

        conv2d = self.conv2d(features).view(-1, self.n_class)
        
        return F.softmax(conv2d, dim=-1)
