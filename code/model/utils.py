import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, accuracy_score
import random
import numpy as np
from torch.utils.data import Dataset
import torch.autograd
import wandb
import scipy
from sklearn.preprocessing import LabelEncoder
import pickle
from torch.utils.data import Sampler
from scipy.stats import pearsonr, spearmanr
#################################################
#------------ Train & Test Function ------------#
#################################################   
def setup_seed(seed):
    #--- Fix random seed ---#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    
def attention_normalize(weights):
    for i in weights.columns:
        W_min = weights[i].min()
        W_max = weights[i].max()
        weights[i] = (weights[i]-W_min)/(W_max-W_min)
    for i in range(weights.shape[0]):
        W_min = weights.iloc[i].min()
        W_max = weights.iloc[i].max()
        weights.iloc[i] = (weights.iloc[i]-W_min)/(W_max-W_min)
    return(weights)




def test_clip(model, device, test_loader):
    model.eval()
    test_loss = 0
    total_acc = 0
    total_matchscore = 0
    total_foscttm = 0
    y_hat_all = []
    y_all = []


    with torch.no_grad():
        for rna, prot in test_loader:

            # Extract features
            RNA_geneID = torch.tensor(rna[:, 1].tolist()).long().to(device)
            Protein_geneID = torch.tensor(prot[:, 1].tolist()).long().to(device)
            rna_input = torch.tensor(rna[:, 3].tolist(), dtype=torch.float32).to(device)
            prot_input = torch.tensor(prot[:, 3].tolist(), dtype=torch.float32).to(device)

            # Forward pass to get embeddings
            rna_embeddings, protein_embeddings = model(
                rna_id=RNA_geneID, 
                rna_x=rna_input, 
                prot_id=Protein_geneID, 
                prot_x=prot_input,
                get_emb=True
            )

            # Compute contrastive loss and similarity metrics
            loss, similarity = CLIPLoss()(rna_embeddings, protein_embeddings)
            acc, matchscore, foscttm = matching_metrics(similarity)

            test_loss += loss.item()
            total_acc += acc
            total_matchscore += matchscore
            total_foscttm += foscttm

    # Average the metrics over the test set
    test_loss /= len(test_loader)
    avg_acc = total_acc / len(test_loader)
    avg_matchscore = total_matchscore / len(test_loader)
    avg_foscttm = total_foscttm / len(test_loader)

    return test_loss, avg_acc, avg_matchscore, avg_foscttm

def matching_metrics(similarity=None, x=None, y=None, **kwargs):
    if similarity is None:
        if x.shape != y.shape:
            raise ValueError("Shapes do not match!")
        similarity = 1 - scipy.spatial.distance_matrix(x, y, **kwargs)
    if not isinstance(similarity, torch.Tensor):
        similarity = torch.from_numpy(similarity)

    with torch.no_grad():
        # similarity = output.logits_per_atac
        batch_size = similarity.shape[0]
        acc_x = (
            torch.sum(
                torch.argmax(similarity, dim=1)
                == torch.arange(batch_size).to(similarity.device)
            )
            / batch_size
        )
        acc_y = (
            torch.sum(
                torch.argmax(similarity, dim=0)
                == torch.arange(batch_size).to(similarity.device)
            )
            / batch_size
        )
        foscttm_x = (
            (similarity > torch.diag(similarity)).float().mean(axis=1).mean().item()
        )
        foscttm_y = (
            (similarity > torch.diag(similarity)).float().mean(axis=0).mean().item()
        )
        # matchscore_x = similarity.softmax(dim=1).diag().mean().item()
        # matchscore_y = similarity.softmax(dim=0).diag().mean().item()
        X = similarity
        mx = torch.max(X, dim=1, keepdim=True).values
        hard_X = (mx == X).float()
        logits_row_sums = hard_X.clip(min=0).sum(dim=1)
        matchscore = hard_X.clip(min=0).diagonal().div(logits_row_sums).mean().item()

        acc = (acc_x + acc_y) / 2
        foscttm = (foscttm_x + foscttm_y) / 2
        # matchscore = (matchscore_x + matchscore_y)/2
        return acc, matchscore, foscttm
    
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits, torch.arange(len(logits), device=logits.device)
    )

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    atac_loss = contrastive_loss(similarity.T)
    return (caption_loss + atac_loss) / 2.0
   
class CLIPLoss(nn.Module):
    def __init__(self, logit_scale=2.6592, requires_grad=False):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.ones([]) * logit_scale, requires_grad=requires_grad
        )

    def forward(self, rna_emb, prot_emb):
        # normalized features
        # rna_emb = rna_emb / rna_emb.norm(dim=-1, keepdim=True)
        # prot_emb = prot_emb / prot_emb.norm(dim=-1, keepdim=True)
        # atac_embeds = atac_embeds / atac_embeds.norm(dim=-1, keepdim=True)
        # rna_embeds = rna_embeds / rna_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_rna= torch.matmul(rna_emb, prot_emb.t()) * logit_scale
        logits_per_prot = logits_per_rna.T

        loss = clip_loss(logits_per_rna)

        return loss, logits_per_rna  # , logits_per_rna
    
def train_clip(args, model, device, train_loader, optimizer, epoch, test_loader=None):
    model.train()
    total_loss = 0
    total_acc = 0
    total_matchscore = 0
    total_foscttm = 0
    test_loss = 0
    test_acc = 0
    test_matchscore = 0
    test_foscttm = 0

    for batch_idx, (rna, prot) in enumerate(train_loader):
        RNA_geneID = torch.tensor(rna[:, 1].tolist()).long().to(device)
        Protein_geneID = torch.tensor(prot[:, 1].tolist()).long().to(device)
        rna_input = torch.tensor(rna[:, 3].tolist(), dtype=torch.float32).to(device)
        prot_input = torch.tensor(prot[:, 3].tolist(), dtype=torch.float32).to(device)

        #--- Prediction ---#
        optimizer.zero_grad()
        rna_embeddings, protein_embeddings = model(
            rna_id=RNA_geneID, 
            rna_x=rna_input, 
            prot_id=Protein_geneID, 
            prot_x=prot_input,
            get_emb = True
        )

        loss, similarity = CLIPLoss()(rna_embeddings, protein_embeddings)
        acc, matchscore, foscttm = matching_metrics(similarity)
     
        #loss = mse_loss
        #loss= contrastive_loss_value
  
        total_loss += loss.item()
        total_acc += acc
        total_matchscore += matchscore
        total_foscttm += foscttm  
       # torch.autograd.set_detect_anomaly(True)

        loss.backward()
        optimizer.step()

         # Log every n batches
        if (batch_idx + 1) % 1000 == 0:
            avg_train_loss = total_loss / (batch_idx + 1)
            avg_acc = total_acc / (batch_idx + 1)
            avg_matchscore = total_matchscore / (batch_idx + 1)
            avg_foscttm = total_foscttm / (batch_idx + 1)
            print('-' * 15)
            print(f'--- Epoch {epoch} Batch {batch_idx + 1} ---', flush=True)
            print('-' * 15)
            print(f'Training set: Average loss: {avg_train_loss:.4f}, Average acc: {avg_acc:.4f}, \
                    Average matchscore: {avg_matchscore:.4f}, Average foscttm: {avg_foscttm:.4f}', flush=True)
            wandb.log({
                'epoch': epoch,
                'batch_idx': batch_idx + 1,
                'train_loss': avg_train_loss,
                'train_acc': avg_acc,
                'train_matchscore': avg_matchscore,
                'train_foscttm': avg_foscttm
            })

             # Perform test and log metrics
            test_loss, test_acc, test_matchscore, test_foscttm = test_clip(model, device, test_loader)
            wandb.log({
                'epoch': epoch,
                'batch_idx': batch_idx + 1,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'test_matchscore': test_matchscore,
                'test_foscttm': test_foscttm    
            })
    print('-' * 15)
    print(f'--- Epoch {epoch} ---', flush=True)
    print('-' * 15)
    print(f'Training set: Average loss: {total_loss:.4f}, Average acc: {total_acc:.4f}, \
            Average matchscore: {total_matchscore:.4f}, Average foscttm: {total_foscttm:.4f}', flush=True)
    print(f'Test set: Average loss: {test_loss:.4f}, Average acc: {test_acc:.4f}, \
            Average matchscore: {test_matchscore:.4f}, Average foscttm: {test_foscttm:.4f}', flush=True)
    return None
    # train_loss /= len(train_loader)
    # train_ccc /= len(train_loader)
    # total_contrastive_loss /= len(train_loader)
    # print('-'*15)
    # print('--- Epoch {} ---'.format(epoch), flush=True)
    # print('-'*15)
    # print('Training set: Average loss: {:.4f}, Average ccc: {:.4f}, \
    #       Contrastive loss: {:.4f}'.format(train_loss, train_ccc, total_contrastive_loss), flush=True)


    # return train_loss, train_ccc




def train_recon(model, device, train_loader, optimizer, epoch):
    model.train()
    prot_train_loss = 0
    rna_train_loss = 0
    
    for batch_idx, (rna, prot) in enumerate(train_loader):
        RNA_geneID = torch.tensor(rna[:, 1].tolist()).long().to(device)
        Protein_geneID = torch.tensor(prot[:, 1].tolist()).long().to(device)
        rna_mask = torch.tensor(rna[:, 4].tolist()).bool().to(device)
        pro_mask = torch.tensor(prot[:, 4].tolist()).bool().to(device)
        rna_input = torch.tensor(rna[:, 0].tolist(), dtype=torch.float32).to(device)
        prot_input = torch.tensor(prot[:, 0].tolist(), dtype=torch.float32).to(device)
        rna_target = torch.tensor(rna[:, 3].tolist(), dtype=torch.float32).to(device)
        prot_target = torch.tensor(prot[:, 3].tolist(), dtype=torch.float32).to(device)

        optimizer.zero_grad()
        # Get the outputs from the model
        rna_out, prot_out = model(
            rna_id=RNA_geneID, 
            rna_x=rna_input, 
            prot_id=Protein_geneID, 
            prot_x=prot_input,
            rna_mask_label=rna_mask,
            prot_mask_label=pro_mask
        )


        rna_mse_loss = F.mse_loss(rna_out, rna_target[rna_mask])
        prot_mse_loss = F.mse_loss(prot_out, prot_target[pro_mask])
        loss = rna_mse_loss + prot_mse_loss

        rna_train_loss += rna_mse_loss.item()
        prot_train_loss += prot_mse_loss.item()

        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 1000 == 0:
            avg_train_prot_loss = prot_train_loss / (batch_idx + 1)
            avg_train_rna_loss = rna_train_loss / (batch_idx + 1)
            print('-' * 15)
            print(f'--- Epoch {epoch} Batch {batch_idx + 1} ---', flush=True)
            print('-' * 15)
            print(f'Training set: Average prot loss: {avg_train_prot_loss:.4f}, Average rna loss: {avg_train_rna_loss:.4f}', flush=True)
            wandb.log({
                'epoch': epoch,
                'batch_idx': batch_idx + 1,
                'train_prot_loss': avg_train_prot_loss,
                'train_rna_loss': avg_train_rna_loss
            })

    prot_train_loss /= len(train_loader)
    rna_train_loss /= len(train_loader)
    print('-' * 15)
    print('--- Epoch {} ---'.format(epoch), flush=True)
    print('-' * 15)
    print('Training set: Average prot loss: {:.4f}, Average rna loss: {:.4f}'.format(prot_train_loss, rna_train_loss), flush=True)

    return prot_train_loss, rna_train_loss



def unpaired_metrics(similarity_matrix, cell_type_rna, cell_type_protein):
    
    # Get most similar CODEX cell for each RNA cell (argmax in similarity matrix)
    rna_to_protein_matches = similarity_matrix.argmax(axis=1)  # Indices of best-matching protein cell

    # Get matched protein cell types
    matched_protein_cell_types = cell_type_protein[rna_to_protein_matches]
    # cell_type_rna = cell_type_rna.cpu().numpy() if torch.is_tensor(cell_type_rna) else np.array(cell_type_rna)
    # matched_protein_cell_types = matched_protein_cell_types.cpu().numpy() if torch.is_tensor(matched_protein_cell_types) else np.array(matched_protein_cell_types)

    # Compute Accuracy (same cell type = correct match)
    accuracy = accuracy_score(cell_type_rna, matched_protein_cell_types)

    # Compute FOSCTTM (Fraction of Same Cell Type in Top Matches)
    foscttm_scores = []

    for i, rna_type in enumerate(cell_type_rna):
        # Get similarity scores for this RNA cell
        similarities = similarity_matrix[i, :]

        # Sort indices by similarity (descending, since higher = better match)
        sorted_indices = np.argsort(-similarities)

        # Find the rank of the correct cell type
        true_match_rank = np.where(cell_type_protein[sorted_indices] == rna_type)[0]

        if len(true_match_rank) > 0:
            # Compute fraction of incorrect matches ranked higher
            foscttm_score = true_match_rank[0] / len(cell_type_protein)
            foscttm_scores.append(foscttm_score)

    return accuracy, np.mean(foscttm_scores)


import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_split_adata(adata, test_size=0.2, random_state=42):
    """
    Splits an AnnData object into train and test sets while stratifying based on cell types.

    Parameters:
    - adata: AnnData object (with cell types in `adata.obs["cell_type"]`)
    - test_size: Fraction of test data (default 0.2)
    - random_state: Random seed for reproducibility (default 42)

    Returns:
    - adata_train: AnnData object for training
    - adata_test: AnnData object for testing
    """
    # Ensure "cell_type" exists in `obs`
    assert "cell_type" in adata.obs, "AnnData object must have 'cell_type' in .obs"

    # Extract indices and cell types
    indices = np.arange(len(adata))
    cell_types = adata.obs["cell_type"].values  # Get cell type labels

    # Stratified split
    train_idx, test_idx = train_test_split(indices, test_size=test_size, stratify=cell_types, random_state=random_state)

    # Subset AnnData
    adata_train = adata[train_idx].copy()
    adata_test = adata[test_idx].copy()

    return adata_train, adata_test



def train(args, model, device, train_loader, optimizer, epoch, test_loader=None):
    model.train()
    loss2 = nn.CosineSimilarity(dim=0, eps=1e-8)
    train_loss = 0
    train_ccc = 0
    train_pearson = 0
    train_spearman = 0
    best_test_loss = float('inf')  # Initialize the best test loss
    
    # add batch index to print the training process
    for batch_idx, (x, y) in enumerate(train_loader):
       
        #--- Extract Feature ---#
        RNA_geneID = torch.tensor(x[:,1].tolist()).long().to(device)
        Protein_geneID = torch.tensor(y[:,1].tolist()).long().to(device)
        rna_mask = torch.tensor(x[:,2].tolist()).bool().to(device)
        pro_mask = torch.tensor(y[:,2].tolist()).bool().to(device)
        x = torch.tensor(x[:,0].tolist(), dtype=torch.float32).to(device)
        y = torch.tensor(y[:,0].tolist(), dtype=torch.float32).to(device)   

        #--- Prediction ---#
        optimizer.zero_grad()
        _,protein_predictions = model(x, RNA_geneID, Protein_geneID, enc_mask=rna_mask, dec_mask=pro_mask)

        #--- Compute Performance Metric ---#
        protein_predictions = torch.squeeze(protein_predictions)
        protein_predictions = torch.where(torch.isnan(y), torch.full_like(protein_predictions, 0), protein_predictions)
        y = torch.where(torch.isnan(y), torch.full_like(y, 0), y)

        mse_loss = F.mse_loss(protein_predictions[pro_mask], y[pro_mask])
        train_loss += mse_loss.item()
        
        # Calculate CCC
        train_ccc += loss2(protein_predictions[pro_mask], y[pro_mask]).item()
        
        # Calculate Pearson and Spearman correlations
        with torch.no_grad():
            pred_flat = protein_predictions[pro_mask].detach().cpu().numpy().flatten()
            target_flat = y[pro_mask].detach().cpu().numpy().flatten()
            
            # Only calculate if we have valid data points
            if len(pred_flat) > 1:
                p_corr, _ = pearsonr(pred_flat, target_flat)
                s_corr, _ = spearmanr(pred_flat, target_flat)
                train_pearson += p_corr
                train_spearman += s_corr
        
        mse_loss.backward()
        optimizer.step()

    # Calculate average metrics
    avg_train_loss = train_loss / (batch_idx + 1)
    avg_train_ccc = train_ccc / (batch_idx + 1)
    avg_train_pearson = train_pearson / (batch_idx + 1)
    avg_train_spearman = train_spearman / (batch_idx + 1)

    print('-' * 15)
    print(f'--- Epoch {epoch} Batch {batch_idx + 1} ---', flush=True)
    print('-' * 15)
    print(f'Training set: Average loss: {avg_train_loss:.4f}, Average ccc: {avg_train_ccc:.4f}, '
          f'Pearson: {avg_train_pearson:.4f}, Spearman: {avg_train_spearman:.4f}', flush=True)
    
    if args.wandb_off == False:
        wandb.log({
            'epoch': epoch,
            'batch_idx': batch_idx + 1,
            'train_loss': avg_train_loss,
            'train_ccc': avg_train_ccc,
            'train_pearson': avg_train_pearson,
            'train_spearman': avg_train_spearman
        })
        
    # Perform test and log metrics
    test_loss, test_ccc, test_pearson, test_spearman, _, _ = test(model, device, test_loader)
    
    if args.wandb_off == False:
        wandb.log({
            'epoch': epoch,
            'batch_idx': batch_idx + 1,
            'test_loss': test_loss,
            'test_ccc': test_ccc,
            'test_pearson': test_pearson,
            'test_spearman': test_spearman
        })

    train_loss /= len(train_loader)
    train_ccc /= len(train_loader)
    train_pearson /= len(train_loader)
    train_spearman /= len(train_loader)
    
    print('-'*15)
    print('--- Epoch {} ---'.format(epoch), flush=True)
    print('-'*15)
    print('Training set: Average loss: {:.4f}, Average ccc: {:.4f}, '
          'Pearson: {:.4f}, Spearman: {:.4f}'.format(
          train_loss, train_ccc, train_pearson, train_spearman), flush=True)
          
    return train_loss, train_ccc, train_pearson, train_spearman
    
def test(model, device, test_loader):
    model.eval()
    loss2 = nn.CosineSimilarity(dim=0, eps=1e-8)
    test_loss = 0
    test_ccc = 0
    test_pearson = 0
    test_spearman = 0
    y_hat_all = []
    y_all = []
    
    with torch.no_grad():
        for x, y in test_loader:
            #--- Extract Feature ---#
            RNA_geneID = torch.tensor(x[:,1].tolist()).long().to(device)
            Protein_geneID = torch.tensor(y[:,1].tolist()).long().to(device)
            rna_mask = torch.tensor(x[:,2].tolist()).bool().to(device)
            pro_mask = torch.tensor(y[:,2].tolist()).bool().to(device)
            x = torch.tensor(x[:,0].tolist(), dtype=torch.float32).to(device)
            y = torch.tensor(y[:,0].tolist(), dtype=torch.float32).to(device)

            #--- Prediction ---#
            _, y_hat = model(x, RNA_geneID, Protein_geneID, enc_mask=rna_mask, dec_mask=pro_mask)

            #--- Compute Performance Metric ---#
            y_hat = torch.squeeze(y_hat)
            y_hat = torch.where(torch.isnan(y), torch.full_like(y_hat, 0), y_hat)
            y = torch.where(torch.isnan(y), torch.full_like(y, 0), y)
            
            # MSE Loss
            test_loss += F.mse_loss(y_hat[pro_mask], y[pro_mask]).item()
            
            # CCC
            test_ccc += loss2(y_hat[pro_mask], y[pro_mask]).item()
            
            # Store predictions and ground truth for batch
            if device == 'cpu':
                pred_batch = y_hat[pro_mask].view(-1).numpy()
                target_batch = y[pro_mask].view(-1).numpy()
            else:
                pred_batch = y_hat[pro_mask].view(-1).detach().cpu().numpy()
                target_batch = y[pro_mask].view(-1).detach().cpu().numpy()
            
            # Calculate Pearson and Spearman for this batch
            if len(pred_batch) > 1:
                p_corr, _ = pearsonr(pred_batch, target_batch)
                s_corr, _ = spearmanr(pred_batch, target_batch)
                test_pearson += p_corr
                test_spearman += s_corr
            
            # Store for returning full predictions
            if device == 'cpu':
                y_hat_all.extend(y_hat[pro_mask].view(y_hat.shape[0], -1).numpy().tolist())
                y_all.extend(y[pro_mask].view(y_hat.shape[0], -1).numpy().tolist())
            else:
                y_hat_all.extend(y_hat[pro_mask].view(y_hat.shape[0], -1).detach().cpu().numpy().tolist())
                y_all.extend(y[pro_mask].view(y_hat.shape[0], -1).detach().cpu().numpy().tolist())
    
    # Average the metrics
    test_loss /= len(test_loader)
    test_ccc /= len(test_loader)
    test_pearson /= len(test_loader)
    test_spearman /= len(test_loader)
    
    print(f'Test set: Average loss: {test_loss:.4f}, Average ccc: {test_ccc:.4f}, '
          f'Pearson: {test_pearson:.4f}, Spearman: {test_spearman:.4f}')
          
    return test_loss, test_ccc, test_pearson, test_spearman, np.array(y_hat_all), np.array(y_all)