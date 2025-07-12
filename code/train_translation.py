import os
import time
import datetime
import argparse
import warnings
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
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

def run_experiment(args, seed, few_shot_frac=None, leave_out_idx=0):
    """Run a single experiment with the given seed and configuration."""
    # Initialize wandb if needed
    if not args.wandb_off:
        import wandb
        run_name = f"{args.exp_name}_{args.dataset}"
        if args.split == 'random':
            run_name += f"_seed{seed}"
        elif args.split == 'few_shot':
            run_name += f"_frac{few_shot_frac}"
        elif args.split == 'leave_out' or args.split == 'few_leave_out':
            run_name += f"_leaveout{leave_out_idx}"
            if args.split == 'few_leave_out':
                run_name += f"_frac{few_shot_frac}"
        
        wandb.init(project="spatialpro", name=run_name, config=vars(args))

    # Update args for current run
    if few_shot_frac is not None:
        args.few_shot_frac = few_shot_frac
    
    print(f"\n=== Running experiment with seed {seed} on dataset {args.dataset} ===")
    if args.split == 'few_leave_out' or args.split == 'leave_out':
        print(f"Leave-out index: {leave_out_idx}")
    if args.split == 'few_shot' or args.split == 'few_leave_out':
        print(f"Few-shot fraction: {args.few_shot_frac}")
    
    # Set seed
    setup_seed(seed)
    
    #########################
    #--- Prepare for CUDA ---#
    #########################
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print(f"use_cuda: {use_cuda}")
    
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
        print(f'Successfully loaded the pretrain model from checkpoints {args.pretrain_rna_checkpoint} and {args.pretrain_pro_checkpoint}')
    
    # Convert the comma-separated string to a list of integers
    device_ids = [int(id) for id in args.device_num.split(',')]
    device = torch.device("cuda", device_ids[0])
    
    # Resume training from breakpoints
    if args.path_checkpoint != None:
        model.load_state_dict(torch.load(args.path_checkpoint), strict=True)
        print(f'Successfully loaded the model from checkpoint {args.path_checkpoint}')
    
    start_epoch = 0
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    #---  Prepare Optimizer ---#
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    #---  Prepare Scheduler ---#
    scheduler = StepLR(optimizer, step_size=args.gamma_step, gamma=args.gamma)
    
    ##########################
    #--- Prepare The Data ---#
    ##########################
    #---  Load Single Cell Data  ---#
    scRNA_adata = sc.read_h5ad(args.RNA_path)
    scP_adata = sc.read_h5ad(args.Pro_path)
    print(f'Original scRNA_adata shape: {scRNA_adata.shape}')
    print(f'Original scP_adata shape: {scP_adata.shape}')
    
    # if args.pretrain_rna_checkpoint != None and args.pretrain_pro_checkpoint != None:
    scRNA_adata = preprocess_entire_dataset(scRNA_adata, args.enc_max_seq_len)
    print(f'scRNA_adata shape after process: {scRNA_adata.shape}')
    scP_adata = preprocess_entire_dataset(scP_adata, args.dec_max_seq_len)
    print(f'scP_adata shape after process: {scP_adata.shape}')
    
    if args.test_only:
        print("Running in test-only mode, skipping training.")
        # Load the model checkpoint if provided
        test_rna = scRNA_adata
        test_protein = scP_adata
        print("Test set size (entire dataset):", scRNA_adata.shape[0])

    else:
        print("Running in training mode, splitting data into train and test sets.")
        # Split data based on specified method
        if args.split == "random":
            print("Splitting data randomly")
            train_index, test_index = next(ShuffleSplit(n_splits=1, test_size=args.frac_finetune_test, random_state=seed).split(scRNA_adata.obs.index))
            print(f'Using random split, # of training samples: {len(train_index)}')
            print(f'Using random split, # of testing samples: {len(test_index)}')
        
        elif args.split == "few_shot":
            print(f"Splitting data for few shot setting with fraction {args.few_shot_frac}")
            train_index, test_index = next(ShuffleSplit(n_splits=1, test_size=args.frac_finetune_test, random_state=seed).split(scRNA_adata.obs.index))
            train_index = train_index[:int(len(train_index) * args.few_shot_frac)]
            print(f'Using few shot setting, # of training samples: {len(train_index)}')
            print(f'Using few shot setting, # of testing samples: {len(test_index)}')
            
        elif args.split == "few_leave_out" or args.split == "leave_out":
            print(f"Splitting data for {'few-' if args.split == 'few_leave_out' else ''}leave-out setting")
            # Original ShuffleSplit for consistency
            orig_train_index, orig_test_index = next(
                ShuffleSplit(n_splits=1, test_size=args.frac_finetune_test, random_state=seed)
                .split(scRNA_adata.obs.index)
            )
            
            # Determine valid cell type column
            valid_celltype_cols = ['celltype', 'celltype.l1']
            celltype_col = next((col for col in valid_celltype_cols if col in scRNA_adata.obs.columns), None)
            
            if celltype_col is None:
                raise ValueError(f"scRNA_adata.obs must contain one of {valid_celltype_cols}")
            
            # Subset the test split for leave-out selection
            adata_test = scRNA_adata[orig_test_index]
            
            # Get the cell types in order of frequency
            cell_types = adata_test.obs[celltype_col].value_counts().index.tolist()
            
            # If leave_out_idx is out of range, use last available index
            if leave_out_idx >= len(cell_types):
                leave_out_idx = len(cell_types) - 1
                
            # Select leave-out cell type
            leave_out_cell_type = cell_types[leave_out_idx]
            print(f"Leaving out cell type: {leave_out_cell_type} for testing")
            
            # Construct train samples from original train, excluding leave-out cell type
            adata_train = scRNA_adata[orig_train_index]
            train_mask = adata_train.obs[celltype_col] != leave_out_cell_type
            filtered_train_idx = adata_train.obs.index[train_mask]
            
            # Apply few-shot reduction if needed
            if args.split == "few_leave_out":
                few_shot_train_index = filtered_train_idx[:int(len(filtered_train_idx) * args.few_shot_frac)]
            else:
                few_shot_train_index = filtered_train_idx
            
            # Construct test samples: only cells of the leave-out type from the test split
            test_index = adata_test.obs.index[adata_test.obs[celltype_col] == leave_out_cell_type].tolist()
            train_index = few_shot_train_index.tolist()
            
            print(f"# of training samples: {len(train_index)}")
            print(f"# of testing samples: {len(test_index)}")
        
        else:
            raise ValueError(f"Unknown split type: {args.split}")
        
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
        train_loader = torch.utils.data.DataLoader(my_trainset, **train_kwargs)
        my_trainset = Joint_Dataset(train_rna, train_protein, args.enc_max_seq_len, args.dec_max_seq_len, dataset_id=args.dataset_id)
        # ###############################
    
    if args.test_only:
        print(" Running in test-only mode, no training will be performed.")
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            test_loss, test_ccc, test_pearson, test_spearman, _, _ = test_translation_(model, device, test_loader)
        print(f'Test Loss: {test_loss:.4f}, Test CCC: {test_ccc:.4f}, Test Pearson: {test_pearson:.4f}, Test Spearman: {test_spearman:.4f}')
        print(f'Test completed in {time.time() - start_time:.2f} seconds')
        return test_loss, test_ccc, test_pearson, test_spearman
    else:
        print("Running in training mode, starting training loop.")
        #---  Training and Testing ---#
        ###############################
        start_time = time.time()
        best_test_ccc = 0
        best_results = {}
        
        for epoch in range(start_epoch+1, args.epochs + 1):
            torch.cuda.empty_cache()
            
            # Train for this epoch and get the best test CCC
            current_ccc, train_metrics, test_metrics = train_translation_(
                args, model, device, train_loader, optimizer, epoch, best_test_ccc, test_loader
            )
            
            # Track metrics
            train_loss, train_ccc, train_pearson, train_spearman = train_metrics
            test_loss, test_ccc, test_pearson, test_spearman = test_metrics
            
            # Update scheduler
            scheduler.step()
            
            # Update best results if we got a better CCC
            if test_ccc > best_test_ccc:
                best_test_ccc = test_ccc
                best_results = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_ccc': train_ccc,
                    'train_pearson': train_pearson,
                    'train_spearman': train_spearman,
                    'test_loss': test_loss,
                    'test_ccc': test_ccc,
                    'test_pearson': test_pearson,
                    'test_spearman': test_spearman,
                    'time': time.time() - start_time,
                    'dataset': args.dataset
                }
                
                # Save best model for this run
                if args.save_dir is not None:
                    model_path = f'/raid/home/yoyowu/scProSpatialcheckpoint/{args.exp_name}_{args.dataset}'
                    if args.split == 'random':
                        model_path += f'_seed{seed}'
                    elif args.split == 'few_shot':
                        model_path += f'_frac{args.few_shot_frac}'
                    elif args.split == 'leave_out' or args.split == 'few_leave_out':
                        model_path += f'_leaveout{leave_out_idx}'
                        if args.split == 'few_leave_out':
                            model_path += f'_frac{args.few_shot_frac}'
                    
                    torch.save(model.state_dict(), f'{model_path}.pth')
        
        # Final output
        print(f'Run completed in {time.time() - start_time:.2f} seconds')
        print(f'Best test results (epoch {best_results["epoch"]}):')
        print(f'  CCC: {best_results["test_ccc"]:.4f}')
        print(f'  Pearson: {best_results["test_pearson"]:.4f}')
        print(f'  Spearman: {best_results["test_spearman"]:.4f}')
        
        # Close the current wandb run
        if not args.wandb_off:
            wandb.finish()
        
        # Add metadata to results
        if args.split == 'random':
            best_results['seed'] = seed
        elif args.split == 'few_shot':
            best_results['few_shot_frac'] = args.few_shot_frac
        elif args.split == 'leave_out' or args.split == 'few_leave_out':
            best_results['leave_out_idx'] = leave_out_idx
            if args.split == 'few_leave_out':
                best_results['few_shot_frac'] = args.few_shot_frac
        
        return best_results

def main():
    parser = argparse.ArgumentParser(description='train translation task')
    parser.add_argument('--exp_name', type=str, default='rna2prot_translation')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for each GPU training (default: 8)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
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
                        help='latent dimension of each token')
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

    parser.add_argument('--path_checkpoint', default="/raid/home/yoyowu/spatialpro/checkpoint/pretrained_model.pth",
                        help='path for loading the resume checkpoint (need specify)')
    parser.add_argument('--dataset_id', type = int, default=2, help='dataset id')
    parser.add_argument('--device_num', type=str, default='5',  help='which cuda to use')
    parser.add_argument('--wandb_off', action='store_true', default=True, help='turn off wandb')
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature for softmax')
    # parser.add_argument('--few_shot_frac', type=float, default=1.0, help='few shot fraction on training')
    parser.add_argument('--dec_gene_emb_file', default=None, help='path for loading the gene embedding')
    parser.add_argument('--enc_gene_emb_file', default=None, help='path for loading the gene embedding')
    parser.add_argument('--split', default='random', help='split method: random, few_shot, leave_out, few_leave_out')
    parser.add_argument('--save_dir', default=None, help='save directory')
    parser.add_argument('--dataset', type=str, default='pbmc',
                        help='Dataset to use: pbmc, liver, or bmmc')
    
    # Add new arguments for multiple experiments
    parser.add_argument('--few_shot_fracs', type=float, nargs='+',
                      default=[0.1, 0.05, 0.02, 0.01, 0.005],
                      help='Fractions of training data to use in few-shot settings')
    parser.add_argument('--leave_out_idxs', type=int, nargs='+',
                      default=[0, 1, 2, 3, 4],
                      help='Indices of cell types to leave out (by frequency order)')
    parser.add_argument('--random_seeds', type=int, nargs='+',
                      default=[1105, 42, 123, 256, 512],
                      help='Random seeds for multiple runs')
    parser.add_argument('--test_only', action='store_true', default=True,
                        help='Run in test-only mode without training')
                      
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    
    # Set dataset paths based on selected dataset
    if args.test_only:
        print("We will use demo dataset for this test_only mode")
        args.RNA_path = '/raid/home/yoyowu/scProSpatial/data/demo_clip_test_rna.h5ad'
        args.Pro_path = '/raid/home/yoyowu/scProSpatial/data/demo_clip_test_protein.h5ad'

    elif args.dataset == 'liver':
        args.RNA_path = '/raid/home/yoyowu/spatialpro/data/CITEseq/liver/rna_w_ID.h5ad'
        args.Pro_path = '/raid/home/yoyowu/spatialpro/data/CITEseq/liver/mar12_protein_w_ID.h5ad'
        
    elif args.dataset == 'bmmc':
        args.RNA_path = '/raid/home/yoyowu/spatialpro/data/CITEseq/BMMC/rna_w_ID.h5ad'
        args.Pro_path = '/raid/home/yoyowu/spatialpro/data/CITEseq/BMMC/mar12_protein_w_ID.h5ad'
            
    elif args.dataset == 'pbmc':
        args.RNA_path = '/raid/home/yoyowu/spatialpro/data/CITEseq/PBMC/feb25_2025_rna_counts.h5ad'
        args.Pro_path = '/raid/home/yoyowu/spatialpro/data/CITEseq/PBMC/mar12_protein_w_ID.h5ad'
            
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Create results directory if it doesn't exist
    results_dir = '/raid/home/yoyowu/spatialpro/demo_rna2prot_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize results storage
    all_results = []
    
    if args.test_only:
        # If test_only is set, run a single test with the first seed
        seed = args.random_seeds[0]
        test_loss, test_ccc, test_pearson, test_spearman = run_experiment(args, seed)
        all_results.append({
        'test_loss': test_loss,
        'test_ccc': test_ccc, 
        'test_pearson': test_pearson,
        'test_spearman': test_spearman
    })
     
    # Handle experiments based on split type
    if args.split == 'random':
        # For random split, run with all seeds
        for seed in args.random_seeds:
            result = run_experiment(args, seed)
            all_results.append(result)
    
    elif args.split == 'few_shot':
        # For few-shot, use first seed and iterate through fractions
        seed = args.random_seeds[0]
        for few_shot_frac in args.few_shot_fracs:
            result = run_experiment(args, seed, few_shot_frac=few_shot_frac)
            all_results.append(result)
    
    elif args.split == 'leave_out':
        # For leave-out, use first seed and iterate through leave-out indices
        seed = args.random_seeds[0]
        for leave_out_idx in args.leave_out_idxs:
            result = run_experiment(args, seed, leave_out_idx=leave_out_idx)
            all_results.append(result)
    
    elif args.split == 'few_leave_out':
        # For few-leave-out, use first seed and iterate through combinations
        seed = args.random_seeds[0]
        few_shot_frac = args.few_shot_fracs[0]  # Use first fraction for leave-out
        for leave_out_idx in args.leave_out_idxs:
            #for few_shot_frac in args.few_shot_fracs:
                result = run_experiment(args, seed, few_shot_frac=few_shot_frac, leave_out_idx=leave_out_idx)
                all_results.append(result)
    
    else:
        raise ValueError(f"Unknown split type: {args.split}")
    
    # Save all results to CSV
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(results_dir, f"{args.exp_name}_{args.dataset}_{args.split}_results.csv")
    results_df.to_csv(results_path, index=False)
    
    if not args.test_only:
       
        # Print summary of best results
        print("\n=== SUMMARY OF RESULTS ===")
        print(f"Dataset: {args.dataset}")
        print(f"Split type: {args.split}")
        print(f"Best test CCC: {results_df['test_ccc'].max():.4f}")
        print(f"Average test CCC: {results_df['test_ccc'].mean():.4f} ± {results_df['test_ccc'].std():.4f}")
        print(f"Best test Pearson: {results_df['test_pearson'].max():.4f}")
        print(f"Average test Pearson: {results_df['test_pearson'].mean():.4f} ± {results_df['test_pearson'].std():.4f}")
        print(f"Best test Spearman: {results_df['test_spearman'].max():.4f}")
        print(f"Average test Spearman: {results_df['test_spearman'].mean():.4f} ± {results_df['test_spearman'].std():.4f}")
    
    # # Get best configuration
    # best_idx = results_df['test_ccc'].idxmax()
    # best_config = results_df.iloc[best_idx]
    # print("\n=== BEST CONFIGURATION ===")
    # for key, value in best_config.items():
    #     print(f"{key}: {value}")

if __name__ == '__main__':
    main()