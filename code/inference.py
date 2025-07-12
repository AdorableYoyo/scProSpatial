import os
import time
import datetime
import argparse
import warnings


import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


import torch.optim as optim
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

import sys 
sys.path.append('code/model') 
from spatialpro_model import *
from utils import *
from trainers import *
from loaders import *
def infer_in_chunks(model, scRNA_adata, scP_adata, device, args, chunk_size=320):
    """
    Process protein predictions in chunks to handle datasets larger than the model's max sequence length.
    
    Args:
        model: The RNA2Prot model
        scRNA_adata: RNA AnnData object
        scP_adata: Protein AnnData object
        device: Torch device
        args: Arguments
        chunk_size: Size of each protein chunk (default: model's dec_max_seq_len)
    
    Returns:
        Complete predictions for all proteins
    """
    total_proteins = scP_adata.shape[1]
    num_chunks = (total_proteins + chunk_size - 1) // chunk_size  # Ceiling division
    print(f"Processing {total_proteins} proteins in {num_chunks} chunks of {chunk_size}")
    
    all_predictions = []
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_proteins)
        
        print(f"Processing chunk {chunk_idx+1}/{num_chunks} (proteins {start_idx}-{end_idx})")
        
        # Create subset of protein data for this chunk
        chunk_scP_adata = scP_adata[:, start_idx:end_idx].copy()
        
        # Create dataset for this chunk
        chunk_dataset = Joint_Dataset(scRNA_adata, chunk_scP_adata, args.enc_max_seq_len, args.dec_max_seq_len, dataset_id=args.dataset_id)
        chunk_loader = torch.utils.data.DataLoader(chunk_dataset, batch_size=args.batch_size, drop_last=False)
        
        # Get predictions for this chunk
        #test_loss, test_ccc, test_pearson, test_spearman, 
        _, _,_,_, chunk_y_hat, _ = test_translation_(model, device, chunk_loader)
        
        all_predictions.append(chunk_y_hat)
    
    # Concatenate all predictions along the protein dimension
    final_predictions = np.concatenate(all_predictions, axis=1)
    return final_predictions
def main():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--exp_name', type=str, default='testcelltypepred',)
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for each GPU training (default: 1)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=2*1e-4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=1, metavar='M',
                        help='Learning rate step gamma (default: 1 (not used))')
    parser.add_argument('--gamma_step', type=float, default=2000,
                        help='Learning rate step (default: 2000 (not used))')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1105,
                        help='random seed (default: 1105)')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--frac_finetune_test', type=float, default=0.1,
                        help='test set ratio')
    parser.add_argument('--dim', type=int, default=128,
                        help='latend dimension of each token')
    parser.add_argument('--enc_max_seq_len', type=int, default=20000,
                        help='sequence length of encoder')
    parser.add_argument('--dec_max_seq_len', type=int, default=320,
                        help='sequence length of decoder')
    parser.add_argument('--translator_depth', type=int, default=2,
                        help='translator depth')
    parser.add_argument('--initial_dropout', type=float, default=0.1,
                        help='sequence length of decoder')
    parser.add_argument('--enc_depth', type=int, default=2,
                        help='sequence length of decoder')
    parser.add_argument('--enc_heads', type=int, default=8,
                        help='sequence length of decoder')
    parser.add_argument('--dec_depth', type=int, default=2,
                        help='sequence length of decoder')
    parser.add_argument('--dec_heads', type=int, default=8,
                        help='sequence length of decoder')
    parser.add_argument('--pretrain_rna_checkpoint', default=None)
    parser.add_argument('--pretrain_pro_checkpoint', default=None,help='path for loading the pretrain protein checkpoint')
    #parser.add_argument('--resume', default=False, help='resume training from breakpoint')
    parser.add_argument('--path_checkpoint', default="/raid/home/yoyowu/spatialpro/checkpoint/pretrained_model.pth",
                        help='path for loading the resume checkpoint (need specify)')
    parser.add_argument('--dataset_id', type = int, default=4, help='dataset id')
    parser.add_argument('--RNA_path', default='/raid/home/yoyowu/spatialpro/data/HTAPP/brst_rna_wID.h5ad',
                        help='path for loading the rna')
    parser.add_argument('--Pro_path', default='/raid/home/yoyowu/spatialpro/prediction/scP_artificial_HTAPP_n_breast.h5ad',
                        help='path for loading the protein')
  
    parser.add_argument('--device_num', type=str, default='5',  help='which cuda to use, if more than one device id specified, turn on multi-gpu training')
    parser.add_argument('--wandb_off', action='store_true', default=True, help='turn off wandb')
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature for softmax')
    parser.add_argument('--few_shot_frac', type=float, default=1.0, help='few shot fraction on training ')
    parser.add_argument('--dec_gene_emb_file', default=None, help='path for loading the gene embedding')
    parser.add_argument('--enc_gene_emb_file', default=None, help='path for loading the gene embedding')
    parser.add_argument('--split', default='random', help='split method')
    parser.add_argument('--save_dir', default="/raid/home/yoyowu/spatialpro/prediction/testxxxxx.h5ad", help='save directory')
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    #########################
    #--- Prepare for DDP ---#
    #########################
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("use_cuda: %s" % use_cuda)
    # ngpus_per_node = torch.cuda.device_count()
    # print("ngpus_per_node: %s" % ngpus_per_node)
    # is_distributed = ngpus_per_node > 1
    print('seed', args.seed)
    setup_seed(args.seed)
    print(torch.__version__)


    if args.wandb_off == False:
        import wandb
        wandb.init(project="spatialpro", name=args.exp_name, config=vars(args))
    ###########################
    #--- Prepare The Model ---#
    ###########################
    model = RNA2Prot(
        dim=args.dim,
        initial_dropout=args.initial_dropout,
        translator_depth=args.translator_depth,
        enc_depth=args.enc_depth,
        enc_heads=args.enc_heads,
        enc_max_seq_len=args.enc_max_seq_len,
        enc_gene_emb_file=args.enc_gene_emb_file,
        dec_depth=args.dec_depth,
        dec_heads=args.dec_heads,
        dec_max_seq_len=args.dec_max_seq_len,
        dec_gene_emb_file=args.dec_gene_emb_file
        )
    
    if args.pretrain_rna_checkpoint != None and args.pretrain_pro_checkpoint != None:
        model.load_pretrained(args.pretrain_rna_checkpoint, args.pretrain_pro_checkpoint)
        print('successfully loaded the pretrain model from checkpoint{} and checkpoint{}'.format(args.pretrain_rna_checkpoint, args.pretrain_pro_checkpoint))
    # Convert the comma-separated string to a list of integers
    device_ids = [int(id) for id in args.device_num.split(',')]
    device = torch.device("cuda", device_ids[0]) 
    #device = torch.device("cuda", args.device_num) if  len(args.device_num) == 1 else torch.device("cuda", args.device_num[0]) 

    # Resume training from breakpoints
    if args.path_checkpoint != None:
        model.load_state_dict(torch.load(args.path_checkpoint),strict=True)
        print('successfully loaded the model from checkpoint{}'.format(args.path_checkpoint))
      
    start_epoch = 0
    model = model.to(device)
    ##########################
    #--- Prepare The Data ---#

    scRNA_adata = sc.read_h5ad(args.RNA_path)
    scP_adata = sc.read_h5ad(args.Pro_path)
    print('original scRNA_adata shape:', scRNA_adata.shape)
    print('original scP_adata shape:', scP_adata.shape)

    setup_seed(args.seed)
    scRNA_adata = preprocess_entire_dataset(scRNA_adata, args.enc_max_seq_len)
    print('scRNA_adata shape after process:', scRNA_adata.shape)
    scP_adata = preprocess_entire_dataset(scP_adata, args.dec_max_seq_len)
    print('scP_adata shape after process:', scP_adata.shape)
    if args.split == 'random':
        print("Splitting data randomly")

        train_index, test_index = next(ShuffleSplit(n_splits=1,test_size=args.frac_finetune_test,random_state=args.seed).split(scRNA_adata.obs.index))
        # print("the test index is {}".format(test_index) ) # for debug
        # print("sample 0.1 from the test index")
        # test_index = test_index[:int(len(test_index)*0.1)]
        # print("the test index is {}".format(test_index) ) # for debug
    
    scRNA_adata = scRNA_adata[test_index]
    scP_adata = scP_adata[test_index]


    my_testset = Joint_Dataset(scRNA_adata, scP_adata, args.enc_max_seq_len, args.dec_max_seq_len, dataset_id=args.dataset_id)
  

    test_loader = torch.utils.data.DataLoader(my_testset, batch_size=args.batch_size, drop_last=False)

   
    # ###############################
    #--- Inferencing---#
    ###############################
    start_time = time.time()
    
    # Check if we need to process in chunks
    total_proteins = scP_adata.shape[1]
    if total_proteins > args.dec_max_seq_len:
        print(f"Total proteins ({total_proteins}) exceeds model's max sequence length ({args.dec_max_seq_len})")
        print(f"Processing proteins in chunks...")
        
        y_hat_all = infer_in_chunks(model, scRNA_adata, scP_adata, device, args, chunk_size=args.dec_max_seq_len)
    else:
        # Original single pass inference
        test_loader = torch.utils.data.DataLoader(my_testset, batch_size=args.batch_size, drop_last=False)
        test_loss, test_ccc, test_pearson, test_spearman, y_hat_all, y_all = test_translation_(model, device, test_loader)
    
    # Check if the number of rows matches
    assert scRNA_adata.n_obs == y_hat_all.shape[0], "Mismatch in number of observations!"
    # Store predictions into AnnData
    scRNA_adata.obsm["protein_predicted"] = y_hat_all
    print("Stored predictions shape:", scRNA_adata.obsm["protein_predicted"].shape)

    predicted_protein_names = scP_adata.var_names.tolist()
    # Make sure lengths match
    assert len(predicted_protein_names) == y_hat_all.shape[1], "Protein name length mismatch!"
    # Store column names (optional, but recommended):
    scRNA_adata.uns["protein_predicted_names"] = predicted_protein_names
    scRNA_adata.uns["protein_predicted_myid"] = scP_adata.var["MYID"].tolist()
    
    # Calculate processing time
    end_time = time.time()
    print(f"Inference completed in {end_time - start_time:.2f} seconds")
    
    scRNA_adata.write_h5ad(args.save_dir)
    print("Saved predictions to AnnData object.")


    
  
if __name__ == '__main__':
    main()