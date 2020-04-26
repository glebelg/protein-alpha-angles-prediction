import torch
import numpy as np

from sklearn.cluster import KMeans


class Clusterizer:
    def __init__(self, dataset, n_clusters=2):
        self.n_clusters = n_clusters
        self.train_alphas = np.array([a for aa in dataset.train_data_features[2] for a in aa])
        
        self.angle_class, self.angle_class_idxs = self._clusterize()
        self.min_max_class = [(min(c), max(c)) for c in self.angle_class]


    def _clusterize(self):
        angle_class_idxs, angle_class = [], []
        
        kmeans = KMeans(n_clusters=self.n_clusters)
        labels = kmeans.fit_predict(np.stack([np.cos(self.train_alphas), np.sin(self.train_alphas)], axis=1))
        
        for i in range(self.n_clusters):
            angle_class_idxs.append(labels == i)
            angle_class.append(self.train_alphas[labels == i])
        
        for i in range(1, self.n_clusters):
            if max(angle_class[i]) >= 3:
                angle_class[i], angle_class[0] = angle_class[0], angle_class[i]
                angle_class_idxs[i], angle_class_idxs[0] = angle_class_idxs[0], angle_class_idxs[i]
                break
                
        return angle_class, angle_class_idxs

    
    def ang_to_class(self, angles):
        seq = angles.flatten()

        for i in range(len(seq)):
            isClass0 = 1
            for j in range(1, self.n_clusters):
                if self.min_max_class[j][0] <= seq[i] <= self.min_max_class[j][1]:
                    seq[i] = j
                    isClass0 = 0
                    break

            seq[i] = 0 if isClass0 else seq[i]
            
        return seq.long()
