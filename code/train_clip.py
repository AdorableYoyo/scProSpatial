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
from sklearn.model_selection import ShuffleSplit

import sys 
sys.path.append('code/model')
from trainers import *
from spatialpro_model import *
from utils import *
from loaders import *

def main():
    parser = argparse.ArgumentParser(description='train clip')
    parser.add_argument('--exp_name', type=str, default='test_clip_seed_reproduce',)
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

    parser.add_argument('--pretrain_rna_checkpoint', default='/raid/home/yoyowu/spatialpro/checkpoint/pretrained_rna_encoder.pth',help='path for loading the pretrain rna checkpoint')
    parser.add_argument('--pretrain_pro_checkpoint', default='/raid/home/yoyowu/spatialpro/checkpoint/pretrained_protein_encoder.pth',help='path for loading the pretrain protein checkpoint')
    parser.add_argument('--dataset_id',  type=int ,default=2, help='dataset id for the dataset')
    parser.add_argument('--path_checkpoint', default=None,
                        help='path for loading the resume checkpoint (need specify)')
    
    parser.add_argument('--RNA_path', default='/raid/home/yoyowu/scProSpatial/data/demo_clip_test_rna.h5ad',
                        help='path for loading the rna')
    parser.add_argument('--Pro_path', default='/raid/home/yoyowu/scProSpatial/data/demo_clip_test_protein.h5ad', 
                        help='path for loading the protein') 
    parser.add_argument('--device_num', type=str, default='4',  help='which cuda to use, if more than one device id specified, turn on multi-gpu training')
    parser.add_argument('--wandb_off', action='store_true', default=True)
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature for softmax')
    parser.add_argument('--split', type=str, default='random', help='split method for training and testing')
    parser.add_argument('--test_only', action='store_true', default=True, help='Run in test-only mode with pretrained models')
    
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("use_cuda: %s" % use_cuda)
    print('seed', args.seed)
    setup_seed(args.seed)
    print(torch.__version__)


    if args.wandb_off == False:
        import wandb
        wandb.init(project="scProSpatial", name=args.exp_name, config=vars(args))

    rna_model = RNA_pretrain(
        dim=args.dim,
        initial_dropout=args.initial_dropout,
        enc_depth=args.enc_depth,
        enc_heads=args.enc_heads,
        enc_max_seq_len=args.enc_max_seq_len
        )
    protein_model = Prot_pretrain(
            dim=args.dim,
            initial_dropout=args.initial_dropout,
            dec_depth=args.dec_depth,
            dec_heads=args.dec_heads,
            dec_max_seq_len=args.dec_max_seq_len
            )

    # Convert the comma-separated string to a list of integers
    device_ids = [int(id) for id in args.device_num.split(',')]
    device = torch.device("cuda", device_ids[0]) 

 
    if args.pretrain_rna_checkpoint != None:
     # Load the checkpoint
        checkpoint = torch.load(args.pretrain_rna_checkpoint)
        rna_model.load_state_dict(checkpoint)
        print(f"Successfully loaded the pre-trained RNA encoder from checkpoint {args.pretrain_rna_checkpoint}")
    if args.pretrain_pro_checkpoint != None:
        checkpoint = torch.load(args.pretrain_pro_checkpoint)
        protein_model.load_state_dict(checkpoint)
        print(f"Successfully loaded the pre-trained Protein encoder from checkpoint {args.pretrain_pro_checkpoint}")
  
    
    start_epoch = 0
    rna_model = rna_model.to(device)
    protein_model = protein_model.to(device)
    
    if len(args.device_num) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    #---  Prepare Optimizer ---#
    optimizer = optim.Adam(list(rna_model.parameters()) + list(protein_model.parameters()), lr=args.lr)
    #---  Prepare Scheduler ---#
    scheduler = StepLR(optimizer, step_size=args.gamma_step, gamma=args.gamma)
 
         
    #---  Load Single Cell Data  ---#
    scRNA_adata = sc.read_h5ad(args.RNA_path)
    scP_adata = sc.read_h5ad(args.Pro_path)
    print('original scRNA_adata shape:', scRNA_adata.shape)
    print('original scP_adata shape:', scP_adata.shape)

    setup_seed(args.seed)
    scRNA_adata = preprocess_entire_dataset(scRNA_adata, args.enc_max_seq_len)
    print('scRNA_adata shape after process:', scRNA_adata.shape)
    scP_adata = preprocess_entire_dataset(scP_adata, args.dec_max_seq_len)
    print('scP_adata shape after process:', scP_adata.shape)



    if args.test_only:
        print("TEST-ONLY mode: Using entire dataset without splitting")
        # Use entire dataset for testing
        test_rna = scRNA_adata
        test_protein = scP_adata
        print("Test set size (entire dataset):", scRNA_adata.shape[0])
    else:
        print("Splitting data")
        train_index, test_index = next(ShuffleSplit(n_splits=1,test_size=args.frac_finetune_test,random_state=args.seed).split(scRNA_adata.obs.index))
        print("the test index is {}".format(test_index))
        
        print("Train set size:", len(train_index))
        print("Test set size:", len(test_index))

        # --- RNA ---#
        train_rna = scRNA_adata[train_index]
        test_rna = scRNA_adata[test_index]
   
      
        # --- Protein ---#
        train_protein = scP_adata[train_index]
        test_protein = scP_adata[test_index]

    
    #---  Construct Dataloader ---#

    train_kwargs = {}
    test_kwargs = {}
    if use_cuda:
        cuda_train_kwargs = {'num_workers': 32,
                       'drop_last': True,
                       'shuffle': True,
                       'batch_size': args.batch_size}
        cuda_test_kwargs = {'num_workers': 32,
                    'drop_last': True,
                    'shuffle': False if args.test_only else True,
                    'batch_size': args.batch_size}
        train_kwargs.update(cuda_train_kwargs)
        test_kwargs.update(cuda_test_kwargs)

    my_testset = Joint_Dataset(test_rna, test_protein, args.enc_max_seq_len, args.dec_max_seq_len, dataset_id=args.dataset_id)
    test_loader = torch.utils.data.DataLoader(my_testset, **test_kwargs)

    if not args.test_only:
        my_trainset = Joint_Dataset(train_rna, train_protein, args.enc_max_seq_len, args.dec_max_seq_len, dataset_id=args.dataset_id)
        train_loader = torch.utils.data.DataLoader(my_trainset, **train_kwargs)

    ###############################
    #---  Training and Testing ---#
    ###############################
    if args.test_only:
        print("Running in TEST-ONLY mode")
        

        rna_model.eval()
        protein_model.eval()
        
        # Run evaluation only
        start_time = time.time()
        with torch.no_grad():
                # perform test and log metrics
            _,test_acc, test_matchscore, test_foscttm = test_clip_sep_encoder(rna_model, protein_model, device,test_loader)
            print(f'Test match score: {test_matchscore}')
            print(f'Test foscttm: {test_foscttm}')
            print(f'Test accuracy: {test_acc}')
            print(f'Test evaluation completed in: {time.time()-start_time:.2f} seconds')
    else:
        print("Running in TRAINING mode")
        start_time = time.time()
        best_test_acc = 0
        for epoch in range(start_epoch+1, args.epochs + 1):
            torch.cuda.empty_cache()
            best_test_acc = train_clip_sep_encoder(rna_model, protein_model, device, train_loader, 
                                test_loader, optimizer, epoch, best_test_acc,
                                args.exp_name)
            scheduler.step()
        print('Total time: ', time.time()-start_time)
        if args.wandb_off == False:
            wandb.finish()
if __name__ == '__main__':
    main()