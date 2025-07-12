import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import Dataset
import torch.autograd

from sklearn.preprocessing import LabelEncoder
import pickle
import scanpy as sc

import torch
import numpy as np
import scanpy as sc
from torch.utils.data import Dataset

def preprocess_entire_dataset(adata, max_seq_len):
    """log transform the dataset if raw counts ."""
    if adata.X.max() > 30:
        # sc.pp.normalize_total(adata, target_sum=1e4)
        # print('applying normalizing total counts with target sum 1e4')
        sc.pp.log1p(adata)
        print('applying log1p')
    return adata


 
class MultiSourceDataset(Dataset):
    """
    For pretraining, we need to handle multiple datasets.
    """
    def __init__(self, adata, dataset_id, max_seq_len, mask_prob=0.3, norm=True):
        super().__init__()
        self.dataset_id = dataset_id
        self.max_seq_len = max_seq_len
        self.adata = self.process_data(adata)
        self.len = self.max_seq_len  # Ensure correct sequence length
        self.mask_prob = mask_prob
        self.norm = norm
        self.gene_ids = self.adata.var['MYID'].astype(int).values.tolist()
        

    def process_data(self, adata):
        print('Processing dataset{}...'.format(self.dataset_id))
        print('Original data shape: {}'.format(adata.shape))
      
        max_value = adata.X.max() if not hasattr(adata.X, 'toarray') else adata.X.toarray().max()
        if max_value > 30:  # Check if raw counts
            print('Found raw counts data from dataset{}. Normalizing...'.format(self.dataset_id))
            sc.pp.normalize_total(adata, target_sum=1e4)
            print('applying normalizing total counts with target sum 1e4')
            sc.pp.log1p(adata)
        if self.max_seq_len > 1000: # proess gene expression data
            sc.pp.highly_variable_genes(adata, n_top_genes=self.max_seq_len)
            adata = adata[:, adata.var.highly_variable]  # Subset to HVGs
        print('Processed data shape: {}'.format(adata.shape))

        return adata

    def __getitem__(self, index):
        cell_data = self.adata[index]  # Select the indexed cell

        # if self.norm:
        #     cell_data = self.normalize(cell_data)  # Apply normalization if needed

        # Normalize, truncate, and apply padding while also handling masking
        masked_values, mask, mask_label, gene_ids, cell_data = self.apply_masking(
            cell_data, self.max_seq_len, self.mask_prob, self.norm
        )
        # masked_values, mask, mask_label = self.apply_masking(cell_data)
        return (
        torch.tensor(masked_values, dtype=torch.float32),
        torch.tensor(gene_ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.bool),
        torch.tensor(cell_data, dtype=torch.float32),
        torch.tensor(mask_label, dtype=torch.bool),
        torch.tensor(self.dataset_id, dtype=torch.long)
    )
    
        # return np.array(masked_values), np.array(self.gene_ids), np.array(mask), np.array(cell_data), np.array(mask_label), dataset_id_token
    
    def __len__(self):
        return self.adata.n_obs

    def normalize(self, x, low=1e-8, high=1):
        MIN, MAX = np.min(x), np.max(x)
        return low + (x - MIN) / (MAX - MIN) * (high - low)
    
    #def apply_masking(self, x):
        # padding mask
    def apply_masking(self, x, length, mask_prob=0.3, norm=True):
        len_x = x.shape[1]  # Number of genes/proteins
        x_values = x.X.A.flatten() if hasattr(x.X, 'A') else x.X.flatten()  # Convert sparse to dense
        if norm:
            x_values = self.normalize(x_values)  # Apply normalization
        if len_x<=length: # apply padding 
           # print('padding')
            gene = x.var['MYID'].astype(int).values.tolist()
            gene.extend([0 for i in range(length-len_x)])
            mask = np.concatenate((np.full(len_x, True), np.full(length-len_x, False)))
            # if len_x == length:
            #     mask = None
            x_values = x_values.tolist() + [0 for i in range(length-len_x)]
        else: 
            # apply truncation
            gene = x.var['MYID'].astype(int).values.tolist()[:length]   
            mask = np.full(length, True)
            x_values = x_values[:length].tolist()
        # applying masking for MLM
        #mask = np.ones(len(x), dtype=bool)  # No padding since HVGs are selected
        mask_indices = np.random.choice(length, int(self.mask_prob * length), replace=False)
        #masked_x = x.copy()
        masked_x = np.array(x_values)
        masked_x[mask_indices] = 0  # Zero represents masked genes
        mask_label = np.zeros_like(x_values)
        mask_label[mask_indices] = 1  # 1 for masked positions
        return masked_x, mask, mask_label, gene, x_values




def normalization(x, low=1e-8, high=1):
    MIN = min(x)
    MAX = max(x)
    x = low + (x-MIN)/(MAX-MIN)*(high-low) # zoom to (low, high)
    return x



def process_data(x, length):
    '''
    x = (num_gene,1)

    '''
        # Convert sparse matrix to dense array
    if hasattr(x.X, "toarray"):  # Check if it's sparse
        x_dense = x.X.toarray().flatten()  # Convert to dense & flatten
    else:
        x_dense = x.X.flatten()  # Already dense
    len_x = len(x_dense)
    #tmp = [i for i in x.X[0]]
    tmp = normalization(x_dense)
    if len_x >= length: # truncate
        x_value = tmp[:length]
        gene = x.var.iloc[:length]['MYID'].astype(int).values.tolist()
        mask = np.full(length, True).tolist()
    else: # padding
        x_value = tmp.tolist()
        x_value.extend([0 for i in range(length-len_x)])
        gene = x.var['MYID'].astype(int).values.tolist()
        gene.extend([0 for i in range(length-len_x)])
        mask = np.concatenate((np.full(len_x,True), np.full(length-len_x,False)))
    return x_value, gene, mask



    
class Joint_Dataset(Dataset):
    def __init__(self, scRNA_adata, scP_adata, len_rna, len_protein, dataset_id = None):
        super().__init__()
        self.scRNA_adata = scRNA_adata
        self.scP_adata = scP_adata
        self.len_rna = len_rna
        self.len_protein = len_protein
        self.dataset_id = dataset_id
        if self.dataset_id is not None:
            print(f'Processing dataset {self.dataset_id}...')

    def __getitem__(self, index):
        k = self.scRNA_adata.obs.index[index]
        rna_value, rna_gene, rna_mask = process_data(self.scRNA_adata[k], self.len_rna)
        pro_value, pro_gene, pro_mask = process_data(self.scP_adata[k], self.len_protein)
        if self.dataset_id is None:
            return np.array([rna_value, rna_gene, rna_mask]), np.array([pro_value, pro_gene, pro_mask])
        return np.array([rna_value, rna_gene, rna_mask]), np.array([pro_value, pro_gene, pro_mask]), self.dataset_id
        # return np.array([rna_value, rna_gene, rna_mask]), np.array([pro_value, pro_gene, pro_mask])

    def __len__(self):
        return self.scRNA_adata.n_obs
 




