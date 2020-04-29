import torch
import warnings
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset


class Dataset:
    def __init__(self, data, max_len_seq, len_subseq, batch_size=16, train_size=0.85):
        assert max_len_seq >= len_subseq + 2, 'max_len_seq must be at least 2 more then len_subseq'

        if len_subseq < 7:
            warnings.warn("some models are not designed for subsequences shorter than 8")
        
        self.data = data
        self.max_len_seq = max_len_seq
        self.len_subseq = len_subseq
        self.batch_size = batch_size
        self.train_size = train_size

        self.train_data_features, self.test_data_features = self._get_train_test_features()
        
        self.train_data, self.test_data = self._get_train_test()

        
    def _get_train_test_features(self):
        seq_ohe_data = []
        pssm_data = []
        alphas_data = []
        
        for i in trange(len(self.data.seqs)):
            if len(self.data.seqs[i]) <= self.max_len_seq:
                cur_seq = np.array([self.data.aa2id[a] for a in self.data.seqs[i]])

                for j in range(1, len(self.data.seqs[i]) - self.len_subseq, self.len_subseq):
                    if np.isnan(self.data.alphas[i][j-1:j+self.len_subseq-2]).sum() == 0:
                        subseq_ohe = F.one_hot(torch.tensor(cur_seq[j:j+self.len_subseq]),len(self.data.aa2id))

                        seq_ohe_data.append(subseq_ohe.type(torch.float))
                        pssm_data.append(torch.tensor(self.data.pssms[i])[:,j:j+self.len_subseq])
                        alphas_data.append(torch.tensor(self.data.alphas[i])[j-1:j+self.len_subseq-2])
                        
        data_features = [torch.stack(seq_ohe_data), torch.stack(pssm_data), torch.stack(alphas_data)]
        
        train_data_features, test_data_features = [], []
        
        for feature in data_features:
            train_data_features.append(feature[:int(feature.shape[0] * self.train_size),...])
            test_data_features.append(feature[int(feature.shape[0] * self.train_size):,...])

        return train_data_features, test_data_features
    

    def _get_train_test(self):
        train_data = DataLoader(TensorDataset(*self.train_data_features), batch_size=self.batch_size, shuffle=True)
        test_data = DataLoader(TensorDataset(*self.test_data_features), batch_size=self.batch_size, shuffle=True)
        return train_data, test_data
    

    def get_sample_batch(self):
        return next(iter(self.train_data))
