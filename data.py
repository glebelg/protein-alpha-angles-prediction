import numpy as np
from tqdm import tqdm, trange


DATA_DIR_PATH = './data/casp12/'
TRAIN_FILES = ['training_30', 'training_50']


class Data:   
    def __init__(self):
        self.ids = []
        self.seqs = []
        self.pssms = []
        self.coords = []
        self.masks = []
        self.alphas = []
        
        print('Files reading...')
        self._read_file()
                
        print('Targets calculating...')
        self._calculate_targs()
        
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa2id = {amino_acids[i] : i for i in range(len(amino_acids))}
        self.id2aa = {i : amino_acids[i] for i in range(len(amino_acids))}
        
    
    def _read_file(self):
        for train_file in TRAIN_FILES:
            print('File:', train_file)
            train_file_path = DATA_DIR_PATH + train_file
            with open(train_file_path) as file:
                lines = file.read().splitlines()
                for i in trange(len(lines)):
                    if lines[i] == "[ID]":
                        id = lines[i+1]
                        i += 1
                    elif lines[i] == "[PRIMARY]":
                        seq = lines[i+1]
                        i += 1
                    elif lines[i] == "[EVOLUTIONARY]":
                        pssm = [[float(c) for c in l.split('\t')] for l in lines[i+1:i+22]]
                        i += 21
                    elif lines[i] == "[TERTIARY]":
                        coord = [[float(c) for c in l.split('\t')] for l in lines[i+1:i+4]]
                        i += 3
                    elif lines[i] == "[MASK]":
                        mask = lines[i+1]
                        i += 1
                    elif len(lines[i]) == 0:
                        if '-' not in mask[1::3]:
                            self.ids.append(id)
                            self.seqs.append(seq)
                            self.pssms.append(pssm)
                            self.coords.append(coord)
                            self.masks.append(mask)


    def _get_dihedral_angle(self, p0, p1, p2, p3):
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2

        b1 /= np.linalg.norm(b1) + 1e-9
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1

        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        angle = np.arctan2(y, x)

        return angle


    def _calculate_targs(self):
        for coord in tqdm(self.coords):
            alpha = []
            p0 = np.array([coord[0][1], coord[1][1], coord[2][1]])
            p1 = np.array([coord[0][4], coord[1][4], coord[2][4]])
            p2 = np.array([coord[0][7], coord[1][7], coord[2][7]])

            for i in range(10, len(coord[0]), 3):
                p3 = np.array([coord[0][i], coord[1][i], coord[2][i]])
                alpha.append(self._get_dihedral_angle(p0, p1, p2, p3))
                p0 = p1
                p1 = p2
                p2 = p3

            self.alphas.append(alpha)
