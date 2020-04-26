import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from IPython.display import clear_output
from sklearn.metrics import f1_score
from tqdm import trange, tqdm


class Trainer:
    def __init__(self, model, dataset, clusterizer):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.clusterizer = clusterizer

        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters())
        
        self.train_loss_log, self.train_f1_log = [], []
        self.test_loss_log, self.test_f1_log = [], []
        self.preds, self.targs = None, None


    def train_epoch(self):
        loss_log, f1_log = [], []
        self.model.train()

        steps = 0

        for (seq, pssm, targ) in tqdm(self.dataset.train_data):
            output = self.model(seq.to(self.device), pssm.to(self.device)).cpu()
            targ = self.clusterizer.ang_to_class(targ)

            loss = self.criterion(output, targ)
            loss_log.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            f1_log.append(f1_score(targ, torch.argmax(output, dim=-1), average='micro'))

            steps += 1

        return loss_log, f1_log, steps


    def test(self):
        loss_log, f1_log = [], []
        preds, targs = [], []
        self.model.eval()

        for (seq, pssm, targ) in tqdm(self.dataset.test_data):
            output = self.model(seq.to(self.device), pssm.to(self.device)).cpu()
            targ = self.clusterizer.ang_to_class(targ)

            preds.append(output)
            targs.append(targ)

            loss = self.criterion(output, targ)
            loss_log.append(loss.item())

            f1_log.append(f1_score(targ, torch.argmax(output, dim=-1), average='micro'))

        self.preds, self.targs = torch.cat(tuple(preds), dim=0).detach(), torch.cat(targs)
        
        return loss_log, f1_log


    def plot_history(self, train_history, val_history, title='loss'):
        plt.figure()
        plt.title('{}'.format(title))
        plt.plot(train_history, label='train', zorder=1)

        points = np.array(val_history)

        plt.scatter(points[:, 0], points[:, 1], marker='+', s=180, c='orange', label='test', zorder=2)
        plt.xlabel('train steps')

        plt.legend(loc='best')
        plt.grid()

        plt.show()


    def train(self, n_epochs):
        for epoch in range(n_epochs):
            print("Epoch {0} of {1}".format(epoch, n_epochs))
            train_loss, train_f1, steps = self.train_epoch()

            test_loss, test_f1 = self.test()

            self.train_loss_log.extend(train_loss)
            self.train_f1_log.extend(train_f1)

            self.test_loss_log.append((steps * (epoch + 1), np.mean(test_loss)))
            self.test_f1_log.append((steps * (epoch + 1), np.mean(test_f1)))

            clear_output()
            self.plot_history(self.train_loss_log, self.test_loss_log)
            self.plot_history(self.train_f1_log, self.test_f1_log, title='f1')
            print("Epoch: {0}, val loss: {1}, val f1: {2}".format(epoch + 1, np.mean(test_loss), np.mean(test_f1)))
