from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
from itertools import cycle
from utils import *
from scipy.stats import pearsonr, spearmanr


def train_clip_sep_encoder(rna_model, protein_model, device, train_loader, 
                               test_loader, optimizer, epoch, best_test_acc, exp_name = None):
    rna_model.train()
    protein_model.train()
    total_loss = 0
    total_acc = 0
    total_foscttm = 0
    total_matchscore = 0
    test_acc = 0
    test_foscttm = 0
    test_matchscore = 0
  
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        if len(batch) == 3:  # (rna_batch, prot_batch, dataset_id)
            rna_batch, prot_batch, data_id = batch
            data_id = torch.tensor(data_id).long().to(device)  # Convert dataset_id to tensor
        else:  # (rna_batch, prot_batch) only
            rna_batch, prot_batch = batch
            data_id = None  # No dataset_id available

        rna_id = torch.tensor(rna_batch[:,1].tolist()).long().to(device)
        prot_id = torch.tensor(prot_batch[:,1].tolist()).long().to(device)
        rna_mask = torch.tensor(rna_batch[:,2].tolist()).bool().to(device)
        prot_mask = torch.tensor(prot_batch[:,2].tolist()).bool().to(device)
        rna_value = torch.tensor(rna_batch[:,0].tolist(), dtype=torch.float32).to(device)
        prot_value = torch.tensor(prot_batch[:,0].tolist(), dtype=torch.float32).to(device)   
        
        # Compute embeddings
        if data_id is not None:
            z_rna = rna_model(rna_id=rna_id, rna_x=rna_value, data_id=data_id, get_emb=True)
            z_protein = protein_model(prot_id=prot_id, prot_x=prot_value, data_id=data_id, get_emb=True, dec_mask=prot_mask)
        else:
            z_rna = rna_model(rna_id=rna_id, rna_x=rna_value, get_emb=True)
            z_protein = protein_model(prot_id=prot_id, prot_x=prot_value, get_emb=True, dec_mask=prot_mask)
      

        # Compute contrastive loss
        loss, similarity = CLIPLoss()(z_rna, z_protein)
        acc, matchscore, foscttm = matching_metrics(similarity)


        total_loss += loss.item()
        total_acc += acc
        total_foscttm += foscttm
        total_matchscore += matchscore
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
    #log evert 800 batches
    #if (batch_idx + 1) % 100 == 0:
    avg_loss = total_loss / (batch_idx + 1)
    avg_acc = total_acc / (batch_idx + 1)
    avg_matchscore = total_matchscore / (batch_idx + 1)
    avg_foscttm = total_foscttm / (batch_idx + 1)
    print(f'Epoch {epoch},  Batch {batch_idx+1}: Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, Matchscore: {avg_matchscore:.4f}, Foscttm: {avg_foscttm:.4f}', flush=True)
    wandb.log({
        'epoch': epoch,
        'batch_idx': batch_idx + 1,
        'train_loss': avg_loss,
        'train_acc': avg_acc,
        'train_foscttm': avg_foscttm,
        'train_matchscore': avg_matchscore
})
        
    # perform test and log metrics
    _,test_acc, test_matchscore, test_foscttm = test_clip_sep_encoder(rna_model, protein_model, device,test_loader)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        wandb.log({'best_test_acc': best_test_acc})
            # save the model 
        
        torch.save(rna_model.state_dict(), '/raid/home/yoyowu/scProSpatial/checkpoint/rna_{}.pth'.format(exp_name))
        torch.save(protein_model.state_dict(), '/raid/home/yoyowu/scProSpatial/checkpoint/protein_{}.pth'.format(exp_name))
        print(f"New best test accuracy: {test_acc:.4f}. Model saved..")
    
    wandb.log({
        'epoch': epoch,
        'batch_idx': batch_idx + 1,
        'test_acc': test_acc,
        'test_foscttm': test_foscttm,
        'test_matchscore': test_matchscore
    })
    print(f'Epoch {epoch}:  Test Acc: {test_acc:.4f}, Test Matchscore: {test_matchscore:.4f}, Test Foscttm: {test_foscttm:.4f}', flush=True)
    
    return best_test_acc

def test_clip_sep_encoder(rna_model, protein_model, device, test_loader):
    rna_model.eval()
    protein_model.eval()

    test_loss = 0
    total_acc = 0
    total_matchscore = 0
    total_foscttm = 0
    # Initialize lists to store all embeddings & labels
 
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if len(batch) == 3:  # (rna_batch, prot_batch, dataset_id)
                rna_batch, prot_batch, data_id = batch
                data_id = torch.tensor(data_id).long().to(device)  # Convert dataset_id to tensor
            else:  # (rna_batch, prot_batch) only
                rna_batch, prot_batch = batch
                data_id = None  # No dataset_id available

            rna_id = torch.tensor(rna_batch[:,1].tolist()).long().to(device)
            prot_id = torch.tensor(prot_batch[:,1].tolist()).long().to(device)
            rna_mask = torch.tensor(rna_batch[:,2].tolist()).bool().to(device)
            prot_mask = torch.tensor(prot_batch[:,2].tolist()).bool().to(device)
            rna_value = torch.tensor(rna_batch[:,0].tolist(), dtype=torch.float32).to(device)
            prot_value = torch.tensor(prot_batch[:,0].tolist(), dtype=torch.float32).to(device)   
    
            # Compute embeddings
            if data_id is not None:
                z_rna = rna_model(rna_id=rna_id, rna_x=rna_value, data_id=data_id, get_emb=True)
                z_protein = protein_model(prot_id=prot_id, prot_x=prot_value, data_id=data_id, get_emb=True, dec_mask=prot_mask)
            else:
                z_rna = rna_model(rna_id=rna_id, rna_x=rna_value, get_emb=True)
                z_protein = protein_model(prot_id=prot_id, prot_x=prot_value, get_emb=True, dec_mask=prot_mask)
      
            loss, similarity = CLIPLoss()(z_rna, z_protein)
            acc, matchscore, foscttm = matching_metrics(similarity)

            test_loss += loss.item()
            total_acc += acc
            total_matchscore += matchscore
            total_foscttm += foscttm
    
    test_loss /=  len(test_loader)
    avg_acc = total_acc / len(test_loader)
    avg_matchscore = total_matchscore / len(test_loader)
    avg_foscttm = total_foscttm / len(test_loader)

    return test_loss, avg_acc, avg_matchscore, avg_foscttm



def train_cell_type_pred(args, model, device, train_loader, optimizer, epoch, test_loader=None):
    model.train()
    total_loss = 0
    all_targets = []
    all_predictions = []
    all_probs = []
    best_pr_auc = float('-inf') # Track the best PR AUC (saved in args)

    # Training loop
    for prot, cell_type in train_loader:
        Protein_geneID = prot[:, 0].long().to(device)
        prot_input = prot[:, 1].float().to(device)
        cell_type = cell_type.to(device)  # Already encoded cell types

        # Forward pass
        optimizer.zero_grad()
        outputs = model(args.cell_state, Protein_geneID, prot_input)

        # Calculate loss (CrossEntropyLoss for classification)
        loss = F.cross_entropy(outputs, cell_type)
        total_loss += loss.item()

        # Store predictions and targets for metric calculations
        _, predicted = torch.max(outputs, 1)
        probs = F.softmax(outputs, dim=1)

        all_targets.extend(cell_type.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().detach().numpy())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # End of epoch logging
    avg_train_loss = total_loss / len(train_loader)
    roc_auc, cell_type_specific_roc_auc = calculate_roc_auc(all_targets, all_probs)
    pr_auc, cell_type_specific_pr_auc = calculate_pr_auc(all_targets, all_probs)
    f1, cell_type_specific_f1 = calculate_f1(all_targets, all_predictions)
    accuracy, cell_type_specific_accuracy = calculate_accuracy(all_targets, all_predictions)

    # Print metrics
    print(f'--- Epoch {epoch} ---', flush=True)
    print(f'Training set: Average loss: {avg_train_loss:.4f}, ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}', flush=True)
    print(f'Cell type-specific ROC AUC: {cell_type_specific_roc_auc}')
    print(f'Cell type-specific PR AUC: {cell_type_specific_pr_auc}')
    print(f'Cell type-specific F1: {cell_type_specific_f1}')
    print(f'Cell type-specific Accuracy: {cell_type_specific_accuracy}')

    # Log metrics to wandb
    wandb.log({
        'epoch': epoch,
        'train_loss': avg_train_loss,
        'train_roc_auc': roc_auc,
        'train_pr_auc': pr_auc,
        'train_f1': f1,
        'train_accuracy': accuracy
    })

    # Save the model if PR AUC improves
    if pr_auc > best_pr_auc:
        best_pr_auc = pr_auc
        model_save_path = "/raid/home/yoyowu/scProSpatial/checkpoint/" + args.exp_name + "_bestprauc.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"New best PR AUC: {pr_auc:.4f}. Model saved to {model_save_path}.")

    # Test evaluation
    if test_loader is not None:
        test_loss, roc_auc_test, pr_auc_test, f1_test, accuracy_test, cell_type_metrics_test = test_cell_type_pred(args, model, device, test_loader)
        wandb.log({
            'epoch': epoch,
            'test_loss': test_loss,
            'test_roc_auc': roc_auc_test,
            'test_pr_auc': pr_auc_test,
            'test_f1': f1_test,
            'test_accuracy': accuracy_test
        })

    return None

# Helper Functions to Calculate Cell-Type Specific Metrics

def calculate_roc_auc(y_true, y_prob):
    try:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        cell_type_specific_roc_auc = roc_auc_score(y_true, y_prob, average=None, multi_class='ovr')
    except ValueError:
        roc_auc = 0.0
        cell_type_specific_roc_auc = [0.0] * len(np.unique(y_true))
    return roc_auc, cell_type_specific_roc_auc


def calculate_pr_auc(y_true, y_probs):
    try:
        pr_auc= average_precision_score(y_true, y_probs, average='weighted')
        cell_type_specific_pr_auc = average_precision_score(y_true, y_probs, average=None)
    except ValueError:
        pr_auc = 0.0
        cell_type_specific_pr_auc = [0.0] * len(np.unique(y_true))
    return pr_auc, cell_type_specific_pr_auc

def calculate_f1(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted')
    cell_type_specific_f1 = f1_score(y_true, y_pred, average=None)
    return f1, cell_type_specific_f1

def calculate_accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    cell_type_specific_accuracy = [accuracy_score(y_true == i, y_pred == i) for i in np.unique(y_true)]
    return accuracy, cell_type_specific_accuracy

def test_cell_type_pred(args,model, device, test_loader):
    model.eval()
    test_loss = 0
    all_targets = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for prot, cell_type in test_loader:
            Protein_geneID = prot[:, 0].long().to(device)
            prot_input = prot[:, 1].float().to(device)
            cell_type = cell_type.to(device)

            # Forward pass
            outputs = model(args.cell_state,Protein_geneID, prot_input)

            # Calculate loss
            loss = F.cross_entropy(outputs, cell_type)
            test_loss += loss.item()

            # Store predictions and targets for metric calculations
            _, predicted = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)

            all_targets.extend(cell_type.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().detach().numpy())

    # Calculate average test loss
    avg_test_loss = test_loss / len(test_loader)

    # Calculate metrics
    roc_auc, cell_type_specific_roc_auc = calculate_roc_auc(all_targets, all_probs)
    pr_auc, cell_type_specific_pr_auc = calculate_pr_auc(all_targets, all_probs)
    f1, cell_type_specific_f1 = calculate_f1(all_targets, all_predictions)
    accuracy, cell_type_specific_accuracy = calculate_accuracy(all_targets, all_predictions)

    # Print metrics
    print(f'Test set: Average loss: {avg_test_loss:.4f}, ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}', flush=True)
    print(f'Cell type-specific ROC AUC: {cell_type_specific_roc_auc}')
    print(f'Cell type-specific PR AUC: {cell_type_specific_pr_auc}')
    print(f'Cell type-specific F1: {cell_type_specific_f1}')
    print(f'Cell type-specific Accuracy: {cell_type_specific_accuracy}')

    # Return metrics
    return avg_test_loss, roc_auc, pr_auc, f1, accuracy, {
        'roc_auc': cell_type_specific_roc_auc,
        'pr_auc': cell_type_specific_pr_auc,
        'f1': cell_type_specific_f1,
        'accuracy': cell_type_specific_accuracy
    }

    
def train_translation_(args, model, device, train_loader, optimizer, epoch, best_test_ccc, test_loader):
    model.train()
    loss2 = nn.CosineSimilarity(dim=0, eps=1e-8)
    train_loss = 0
    train_ccc = 0
    train_pearson = 0
    train_spearman = 0
    
    # add batch index to print the training process
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        if len(batch) == 3:  # (rna_batch, prot_batch, dataset_id)
            rna_batch, prot_batch, data_id = batch
            data_id = torch.tensor(data_id).long().to(device)  # Convert dataset_id to tensor
        else:  # (rna_batch, prot_batch) only
            rna_batch, prot_batch = batch
            data_id = None  # No dataset_id available
        
        # Get embeddings
        rna_id = torch.tensor(rna_batch[:,1].tolist()).long().to(device)
        prot_id = torch.tensor(prot_batch[:,1].tolist()).long().to(device)
        rna_mask = torch.tensor(rna_batch[:,2].tolist()).bool().to(device)
        prot_mask = torch.tensor(prot_batch[:,2].tolist()).bool().to(device)
        rna_value = torch.tensor(rna_batch[:,0].tolist(), dtype=torch.float32).to(device)
        prot_value = torch.tensor(prot_batch[:,0].tolist(), dtype=torch.float32).to(device)   

        #--- Prediction ---#
        if data_id is not None:
            protein_predictions = model(rna_id, rna_value, prot_id, data_id=data_id, enc_mask=rna_mask, dec_mask=prot_mask) 
        else:
            protein_predictions = model(rna_id, rna_value, prot_id, enc_mask=rna_mask, dec_mask=prot_mask)
            
        #--- Compute Performance Metric ---#
        protein_predictions = torch.where(torch.isnan(prot_value), torch.full_like(protein_predictions, 0), protein_predictions)
        prot_value = torch.where(torch.isnan(prot_value), torch.full_like(prot_value, 0), prot_value)

        mse_loss = F.mse_loss(protein_predictions[prot_mask], prot_value[prot_mask])
        train_loss += mse_loss.item()
        
        # Calculate CCC
        train_ccc += loss2(protein_predictions[prot_mask], prot_value[prot_mask]).item()
        
        # Calculate Pearson and Spearman correlations
        with torch.no_grad():
            pred_flat = protein_predictions[prot_mask].detach().cpu().numpy().flatten()
            target_flat = prot_value[prot_mask].detach().cpu().numpy().flatten()
            
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
    print(f'--- Epoch {epoch} ---', flush=True)
    print('-' * 15)
    print(f'Training set: Average loss: {avg_train_loss:.4f}, Average ccc: {avg_train_ccc:.4f}, '
          f'Pearson: {avg_train_pearson:.4f}, Spearman: {avg_train_spearman:.4f}', flush=True)
    
    if not args.wandb_off:
        wandb.log({
            'epoch': epoch,
            'batch_idx': batch_idx + 1,
            'train_loss': avg_train_loss,
            'train_ccc': avg_train_ccc,
            'train_pearson': avg_train_pearson,
            'train_spearman': avg_train_spearman
        })
    
    # Perform test and log metrics
    test_loss, test_ccc, test_pearson, test_spearman, _, _ = test_translation_(model, device, test_loader)
    
    if not args.wandb_off:
        wandb.log({
            'epoch': epoch,
            'batch_idx': batch_idx + 1,
            'test_loss': test_loss,
            'test_ccc': test_ccc,
            'test_pearson': test_pearson,
            'test_spearman': test_spearman
        })
    
    print(f'Test set: Average loss: {test_loss:.4f}, Average ccc: {test_ccc:.4f}, '
          f'Pearson: {test_pearson:.4f}, Spearman: {test_spearman:.4f}', flush=True)
    
    # Save the model if this is the best test CCC so far
    if test_ccc > best_test_ccc:
        best_test_ccc = test_ccc
        if not args.wandb_off:
            wandb.log({'best_test_ccc': best_test_ccc})
        if args.save_dir is not None:
            torch.save(model.state_dict(), f"{args.save_dir}/{args.exp_name}_best_model.pth")
        print(f"New best model saved with test ccc: {test_ccc:.4f}")
    
    # Return train and test metrics
    train_metrics = (avg_train_loss, avg_train_ccc, avg_train_pearson, avg_train_spearman)
    test_metrics = (test_loss, test_ccc, test_pearson, test_spearman)
    
    return best_test_ccc, train_metrics, test_metrics

def test_translation_(model, device, test_loader):
    model.eval()
    loss2 = nn.CosineSimilarity(dim=0, eps=1e-8)
    test_loss = 0
    test_ccc = 0
    test_pearson = 0
    test_spearman = 0
    y_hat_all = []
    y_all = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if len(batch) == 3:  # (rna_batch, prot_batch, dataset_id)
                rna_batch, prot_batch, data_id = batch
                data_id = torch.tensor(data_id).long().to(device)  # Convert dataset_id to tensor
            else:  # (rna_batch, prot_batch) only
                rna_batch, prot_batch = batch
                data_id = None  # No dataset_id available
            
            #--- Extract Feature ---#
            rna_id = torch.tensor(rna_batch[:,1].tolist()).long().to(device)
            prot_id = torch.tensor(prot_batch[:,1].tolist()).long().to(device)
            rna_mask = torch.tensor(rna_batch[:,2].tolist()).bool().to(device)
            prot_mask = torch.tensor(prot_batch[:,2].tolist()).bool().to(device)
            rna_value = torch.tensor(rna_batch[:,0].tolist(), dtype=torch.float32).to(device)
            prot_value = torch.tensor(prot_batch[:,0].tolist(), dtype=torch.float32).to(device)   

            #--- Prediction ---#
            if data_id is not None:
                y_hat = model(rna_id, rna_value, prot_id, data_id=data_id, enc_mask=rna_mask, dec_mask=prot_mask) 
            else:
                y_hat = model(rna_id, rna_value, prot_id, enc_mask=rna_mask, dec_mask=prot_mask)

            #--- Compute Performance Metric ---#
            y_hat = torch.squeeze(y_hat)
            y_hat = torch.where(torch.isnan(prot_value), torch.full_like(y_hat, 0), y_hat)
            prot_value = torch.where(torch.isnan(prot_value), torch.full_like(prot_value, 0), prot_value)
            
            test_loss += F.mse_loss(y_hat[prot_mask], prot_value[prot_mask]).item()
            test_ccc += loss2(y_hat[prot_mask], prot_value[prot_mask]).item()
            
            # Calculate Pearson and Spearman for batch
            pred_flat = y_hat[prot_mask].detach().cpu().numpy().flatten()
            target_flat = prot_value[prot_mask].detach().cpu().numpy().flatten()
            
            if len(pred_flat) > 1:
                p_corr, _ = pearsonr(pred_flat, target_flat)
                s_corr, _ = spearmanr(pred_flat, target_flat)
                test_pearson += p_corr
                test_spearman += s_corr

            # Store predictions and targets
            if device == 'cpu':
                y_hat_all.extend(y_hat[prot_mask].view(y_hat.shape[0], -1).numpy().tolist())
                y_all.extend(prot_value[prot_mask].view(y_hat.shape[0], -1).numpy().tolist())
            else:
                y_hat_all.extend(y_hat[prot_mask].view(y_hat.shape[0], -1).detach().cpu().numpy().tolist())
                y_all.extend(prot_value[prot_mask].view(y_hat.shape[0], -1).detach().cpu().numpy().tolist())
       
    # Calculate average metrics
    test_loss /= len(test_loader)
    test_ccc /= len(test_loader)
    test_pearson /= len(test_loader)
    test_spearman /= len(test_loader)
    
    return test_loss, test_ccc, test_pearson, test_spearman, np.array(y_hat_all), np.array(y_all)

