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
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Batch, Data, Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
from torch_geometric.utils import to_dense_batch


import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem import rdPartialCharges

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import warnings
warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')

def read_data1(data_path):
    df_read = pq.read_table(data_path).to_pandas()
    lst = df_read.values.tolist()

    return lst

def read_data2(data_path):
    data_1 = pd.read_csv(data_path)
    # MoleculeACE
    data_smiles = data_1.loc[data_1['split']=='train', 'smiles'].tolist()
    bioactivity = np.array(data_1.loc[data_1['split']=='train', 'y'].tolist()) + 9

    return data_smiles, bioactivity


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
        # Atom type
        atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
        # Degree
        n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
        # Implicit valence
        formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
        # Hybridization
        hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
        
        # is in a ring
        is_in_a_ring_enc = [int(atom.IsInRing())]
        # is aromatic
        is_aromatic_enc = [int(atom.GetIsAromatic())]
        # Atomic mass, vdw radius, covalent radius
        atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
        # Vdw radius
        vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
        # Covalent radius
        covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]

        # Chirality
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

class AC_Dataset(Dataset):
    def __init__(self, data_path):
        super(ACDataset, self).__init__()
        self.data = read_data1(data_path)
        self.len_data = len(self.data)

    def get(self, index):
    
        anchor_smile = self.data[index][0]
        anchor_pki = self.data[index][7]

        random.seed(time.perf_counter())
        if len(self.data[index][4]) != 0:
            AC_index = random.choice(range(len(self.data[index][4])))
            AC_smile = self.data[index][4][AC_index]
            AC_pki = self.data[index][6][AC_index]
        else:
            AC_index = random.choice(range(self.len_data))
            AC_smile = self.data[AC_index][0]
            AC_pki = self.data[AC_index][7]

        if len(self.data[index][3]) != 0:
            nonAC_index = random.choice(range(len(self.data[index][3])))
            nonAC_smile = self.data[index][3][nonAC_index]
            nonAC_pki = self.data[index][5][nonAC_index]
        else:
            nonAC_smile = anchor_smile
            nonAC_pki = anchor_pki
        try:
            x1, mol_edge_index1, mol_edge_attr1 = get_feature(anchor_smile)
            x2, mol_edge_index2, mol_edge_attr2 = get_feature(AC_smile)
            x3, mol_edge_index3, mol_edge_attr3 = get_feature(nonAC_smile)
        except:
            print('error：', index)

        
        graph_anchor = Data(x=x1, edge_index=mol_edge_index1, edge_attr=mol_edge_attr1, anchor_pki = anchor_pki)
        graph_AC = Data(x=x2, edge_index=mol_edge_index2, edge_attr=mol_edge_attr2, AC_pki = AC_pki)
        graph_nonAC = Data(x=x3, edge_index=mol_edge_index3, edge_attr=mol_edge_attr3, nonAC_pki = nonAC_pki)
        
        mol_list = []
        mol_list.append(graph_anchor)
        mol_list.append(graph_AC)
        mol_list.append(graph_nonAC)
        return mol_list

    def len(self):
        return self.len_data

class Bio_AC_Dataset(Dataset):
    def __init__(self, data_path):
        super(ActivityDataset, self).__init__()

        self.data = read_data1(data_path)
        self.len_data = len(self.data)

    def get(self, index):
    
        anchor_smile = self.data[index][0]
        anchor_pki = self.data[index][7]

        random.seed(time.perf_counter())
        if len(self.data[index][4]) != 0:
            AC_index = random.choice(range(len(self.data[index][4])))
            AC_smile = self.data[index][4][AC_index]
            AC_pki = self.data[index][6][AC_index]
        else:
            AC_index = random.choice(range(self.len_data))
            AC_smile = self.data[AC_index][0]
            AC_pki = self.data[AC_index][7]

        if len(self.data[index][3]) != 0:
            nonAC_index = random.choice(range(len(self.data[index][3])))
            nonAC_smile = self.data[index][3][nonAC_index]
            nonAC_pki = self.data[index][5][nonAC_index]
        else:
            nonAC_smile = anchor_smile
            nonAC_pki = anchor_pki
        
        if len(self.data[index][4]) != 0 and len(self.data[index][3]) != 0:
            x2, mol_edge_index2, mol_edge_attr2 = get_feature(anchor_smile)
            x3, mol_edge_index3, mol_edge_attr3 = get_feature(AC_smile)
            x2_pki = anchor_pki
            x3_pki = AC_pki
        elif len(self.data[index][3]) == 0:
            x2, mol_edge_index2, mol_edge_attr2 = get_feature(anchor_smile)
            x3, mol_edge_index3, mol_edge_attr3 = get_feature(AC_smile)
            x2_pki = anchor_pki
            x3_pki = AC_pki
        else:
            x2, mol_edge_index2, mol_edge_attr2 = get_feature(anchor_smile)
            x3, mol_edge_index3, mol_edge_attr3 = get_feature(nonAC_smile)
            x2_pki = anchor_pki
            x3_pki = nonAC_pki

        if x2_pki > x3_pki:
            graph_high = Data(x=x2, edge_index=mol_edge_index2, edge_attr=mol_edge_attr2, high_pki = x2_pki)
            graph_low = Data(x=x3, edge_index=mol_edge_index3, edge_attr=mol_edge_attr3, low_pki = x3_pki)
        else:
            graph_high = Data(x=x3, edge_index=mol_edge_index3, edge_attr=mol_edge_attr3, high_pki = x3_pki)
            graph_low = Data(x=x2, edge_index=mol_edge_index2, edge_attr=mol_edge_attr2, low_pki = x2_pki)

        mol_list = []
        mol_list.append(graph_high)
        mol_list.append(graph_low)
        return mol_list

    def len(self):
        return self.len_data

        
class ActivityDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, data_path):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_data_loaders(self):
        train_dataset = Bio_AC_Dataset(data_path=self.data_path)
        train_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))

        random_state = np.random.RandomState(seed=2023)
        random_state.shuffle(indices)

        train_idx = indices
        train_sampler = SubsetRandomSampler(train_idx)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                num_workers=self.num_workers, drop_last=False, pin_memory=True, persistent_workers=True)

        return train_loader
    
class ACDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size

    def get_data_loaders(self):
        train_dataset = AC_Dataset(data_path=self.data_path)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, len(train_dataset)

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))

        random_state = np.random.RandomState(seed=2023)
        random_state.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices, indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                num_workers=self.num_workers, drop_last=False, pin_memory=True, persistent_workers=True)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                num_workers=self.num_workers, drop_last=False, pin_memory=True, persistent_workers=True)

        return train_loader, valid_loader

if __name__ == "__main__":
    dataset = ACDatasetWrapper(4, 8, 0.1, '/home/xxx/SCL_AC/data_pre/Downstream_dataset/cm236_AC_0918.parquet')

    train_loader, _, _ = dataset.get_data_loaders()
    for bn_num, a in enumerate(train_loader):
        if bn_num == 9:

            print(a[1])
            print(a[2])
            print(a[3])
            print(a[4])
