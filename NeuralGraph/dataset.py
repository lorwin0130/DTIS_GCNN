import torch as T
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from . import feature
from . import preprocessing as prep
from . import processing as pro


class MolData(Dataset):
    """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""
    def __init__(self, smiles, labels):
        self.max_atom = 80
        self.max_degree = 6
        self.atoms, self.bonds, self.edges = self._featurize(smiles)
        self.label = T.from_numpy(labels).float()

    def _featurize(self, smiles):
        return prep.tensorise_smiles(smiles, max_atoms=self.max_atom, max_degree=self.max_degree)

    def __getitem__(self, i):
        return self.atoms[i], self.bonds[i], self.edges[i], self.label[i]

    def split(self, batch_size):
        return

    def __len__(self):
        return len(self.label)


class AllData(Dataset):
    def __init__(self, pd_lst, data_path):
        self.max_atom = 80
        self.max_degree = 6
        self.max_atom_p = 300
        self.max_degree_p = 15
        self.atoms, self.bonds, self.edges, self.node, self.verge, self.label = self._featurize(pd_lst, data_path)

    def _featurize(self, pd_lst, data_path):
        return pro.pd_to_input(pd_lst, data_path, max_degree=self.max_degree, max_atoms=self.max_atom, max_degree_p=self.max_degree_p, max_atoms_p=self.max_atom_p)

    def __getitem__(self, i):
        return self.atoms[i], self.bonds[i], self.edges[i], self.node[i], self.verge[i], self.label[i]

    def split(self, batch_size):
        return

    def __len__(self):
        return len(self.label)


class AllData_pk(Dataset):
    def __init__(self, out):
        self.atoms, self.bonds, self.edges, self.node, self.verge, self.label = self._featurize(out)

    def _featurize(self, out):
        return out

    def __getitem__(self, i):
        return self.atoms[i], self.bonds[i], self.edges[i], self.node[i], self.verge[i], self.label[i]

    def split(self, batch_size):
        return

    def __len__(self):
        return len(self.label)