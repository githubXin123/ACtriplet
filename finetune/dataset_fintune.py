import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
import pickle

from copy import deepcopy

import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, RandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Batch, Data, Dataset, DataLoader

from prefetch_generator import BackgroundGenerator


import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem import rdPartialCharges

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import warnings
warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')

def read_data(datalist, labellist):
    smiles_train = datalist
    y_train = labellist
    # y_train = np.array(labellist) + 9

    return smiles_train, y_train

def read_prot(prot_file):
    protein_path = os.path.join('/home/xxyu/SCL_AC/data_pre/protein_emb', prot_file)
    protein_feat = torch.tensor(np.loadtxt(protein_path, delimiter=','))
    return protein_feat

permitted_list_of_atoms = [
            "B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S", "Si", 'Se', 'Te'
        ]

permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE,
                                    Chem.rdchem.BondType.DOUBLE,
                                    Chem.rdchem.BondType.TRIPLE,
                                    Chem.rdchem.BondType.AROMATIC]

def one_hot_encoding(x, permitted_list):
    """
    Creates a binary one-hot encoding of x with respect to the elements in permitted_list. Identifies an input element x that is not in permitted_list with the last element of permitted_list.
    """

    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

    return binary_encoding

def sigmoid(number: float):
    """ numerically semi-stable sigmoid function to map charge between 0 and 1 """
    return 1.0 / (1.0 + float(np.exp(-number)))


def get_feature(smi):
    mol = Chem.MolFromSmiles(smi)
    xs = []
    for atom in mol.GetAtoms():
        atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
        n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
        formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
        hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
        
        is_in_a_ring_enc = [int(atom.IsInRing())]
        is_aromatic_enc = [int(atom.GetIsAromatic())]
        atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
        vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
        covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]

        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])

        atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc \
                                + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc \
                                + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled \
                                + chirality_type_enc + n_hydrogens_enc

        x = torch.tensor(atom_feature_vector)
        xs.append(x)

    mol_x = torch.stack(xs, dim=0)

    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]

        permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE,
                                    Chem.rdchem.BondType.DOUBLE,
                                    Chem.rdchem.BondType.TRIPLE,
                                    Chem.rdchem.BondType.AROMATIC]

        bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
        bond_is_conj_enc = [int(bond.GetIsConjugated())]
        bond_is_in_ring_enc = [int(bond.IsInRing())]
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])

        bond_feature = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc + stereo_type_enc

        edge_attrs.append(bond_feature)
    edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)
    mol_edge_attr = torch.cat([edge_attrs, edge_attrs], dim = 0)

    edge_indices = torch.tensor(edge_indices, dtype=torch.long).view(-1,2)
    edge_indices = torch.cat([edge_indices, edge_indices[:, [1, 0]]], dim=0)

    mol_edge_index = edge_indices.t().contiguous()

    return mol_x, mol_edge_index, mol_edge_attr

class ftDataset(Dataset):
    def __init__(self, datalist, labellist, prot_file):
        super(ftDataset, self).__init__()
        self.smiles, self.real_label = read_data(datalist, labellist)
        self.len_data = len(self.smiles)

    def get(self, index):
    
        smile = self.smiles[index]
        try:
            x1, mol_edge_index1, mol_edge_attr1 = get_feature(smile)
        except:
            print('error：', index)


        real_label = torch.tensor(self.real_label[index], dtype=torch.float).view(1,-1)
        
        graph_smile = Data(x=x1, edge_index=mol_edge_index1, edge_attr=mol_edge_attr1, 
                            y = real_label)
        
        return graph_smile

    def len(self):
        return self.len_data

class ftDatasetWrapper(object):
    def __init__(self, batchsize_train, batchsize_valid, num_workers, train_list, train_y_list, val_list, val_y_list, prot_file):
        super(object, self).__init__()
        self.train_list = train_list
        self.train_y_list = train_y_list
        self.valid_list = val_list
        self.valid_y_list = val_y_list
        self.prot_file = prot_file
        self.batchsize_train = batchsize_train
        self.batchsize_valid = batchsize_valid

        self.num_workers = num_workers


    def get_data_loaders(self):
        train_dataset = ftDataset(self.train_list, self.train_y_list, self.prot_file)
        valid_dataset = ftDataset(self.valid_list, self.valid_y_list, self.prot_file)
        train_loader = self.get_train_validation_data_loaders(train_dataset, self.batchsize_train)
        valid_loader = self.get_train_validation_data_loaders(valid_dataset, self.batchsize_valid)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset, batch_size):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        print('number of training data:', num_train)

        random_state = np.random.RandomState(seed=2023)
        random_state.shuffle(indices)

        train_idx = indices

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)


        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                num_workers=self.num_workers, drop_last=False, pin_memory=True, persistent_workers=True)

        return train_loader
    
class testDatasetWrapper(object):
    def __init__(self, batchsize_test, num_workers, test_list, test_y_list, prot_file):
        super(object, self).__init__()
        self.test_list = test_list
        self.test_y_list = test_y_list
        self.prot_file = prot_file
        self.batchsize_test = batchsize_test

        self.num_workers = num_workers


    def get_data_loaders(self):
        test_dataset = ftDataset(self.test_list, self.test_y_list, self.prot_file)
        test_loader = self.get_train_validation_data_loaders(test_dataset, self.batchsize_test)
        return test_loader

    def get_train_validation_data_loaders(self, train_dataset, batch_size):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        print('number of test data:', num_train)

        indices = list(range(num_train))

        # define samplers for obtaining training and validation batches
        train_sampler = SequentialSampler(indices)


        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                num_workers=self.num_workers, drop_last=False, pin_memory=True, persistent_workers=True)

        return train_loader

if __name__ == "__main__": 

    from MoleculeACE import Data as ACdata
    from sklearn.model_selection import StratifiedKFold 

    dataset_name = 'CHEMBL1871_Ki'
    data_1 = ACdata(dataset_name)
    ss = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    cutoff = np.median(data_1.y_train)
    labels = [0 if i < cutoff else 1 for i in data_1.y_train]
    splits = [{'train_idx': i, 'val_idx': j} for i, j in ss.split(labels, labels)]

    split = splits[0]
    x_tr_fold = [data_1.smiles_train[i] for i in split['train_idx']]
    y_tr_fold = [data_1.y_train[i] for i in split['train_idx']]
    x_val_fold = [data_1.smiles_train[i] for i in split['val_idx']]
    y_val_fold = [data_1.y_train[i] for i in split['val_idx']]

    data_load = ftDatasetWrapper(4, 4, 1, x_tr_fold, y_tr_fold, x_val_fold, y_val_fold)

    train_loader, valid_loader = data_load.get_data_loaders()

    for i in range(3):
        for bn_num, a in enumerate(train_loader):
            print(a)
            print(len(a))
            break






