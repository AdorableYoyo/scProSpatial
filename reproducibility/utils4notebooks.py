import pickle
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import squidpy as sq
import os 

import matplotlib.pyplot as plt
# Note: geomloss needs to be installed separately: pip install geomloss
# import geomloss  # Uncomment when available


def load_embeddings(rna_path, protein_path):
    """
    Load RNA and protein embeddings from pickle files.
    
    Parameters:
    -----------
    rna_path : str
        Path to the RNA embeddings pickle file
    protein_path : str
        Path to the protein embeddings pickle file
        
    Returns:
    --------
    tuple
        (rna_embeddings, protein_embeddings) as numpy arrays
    """
    with open(rna_path, 'rb') as handle:
        rna_embeddings = pickle.load(handle)
    with open(protein_path, 'rb') as handle:
        protein_embeddings = pickle.load(handle)
    
    print(f"RNA embeddings shape: {rna_embeddings.shape}")
    print(f"Protein embeddings shape: {protein_embeddings.shape}")
    
    return rna_embeddings, protein_embeddings


def compute_spot_level_predictions(rna_embeddings, protein_embeddings, scRNA_pred_path, k=50, normalize=False):
    """
    Compute spot-level protein predictions using RNA-protein alignment.
    
    Parameters:
    -----------
    rna_embeddings : numpy.ndarray
        RNA embeddings matrix
    protein_embeddings : numpy.ndarray
        Protein embeddings matrix
    scRNA_pred_path : str
        Path to scRNA anndata with predicted proteins
    k : int, default=50
        Number of top matches to consider per spot
    
     
      
       
        
         
           : bool, default=False
        Whether to normalize embeddings before computing similarity
        
    Returns:
    --------
    tuple
        (predicted_protein_at_spots, scRNA_pred) - torch tensor of predictions and anndata object
    """
    # Convert to torch tensors
    rna_emb = torch.tensor(rna_embeddings)
    codex_emb = torch.tensor(protein_embeddings)
    
    # # Optionally normalize embeddings
    # if normalize:
    #     rna_emb = F.normalize(rna_emb, dim=1)
    #     codex_emb = F.normalize(codex_emb, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(codex_emb, rna_emb.T)
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    
    # Get top-k matches per spot
    topk_scores, topk_indices = sim_matrix.topk(k=k, dim=1)
    print(f"Top-k scores shape: {topk_scores.shape}, Top-k indices shape: {topk_indices.shape}")
    
    # Apply softmax to get weights
    weights = F.softmax(topk_scores, dim=1)
    print(f"Weights shape: {weights.shape}")
    
    # Load scRNA predictions
    scRNA_pred = sc.read_h5ad(scRNA_pred_path)
    pred_np = scRNA_pred.obsm["protein_predicted"]
    # # normalize predicted proteins if needed
    if normalize:
        print("Normalizing predicted protein data...")
        pred_np = normalize_data(pred_np)
    print(f"Predicted proteins shape: {pred_np.shape}")
    
    # Gather the top-k predicted protein values
    topk_protein_preds = pred_np[topk_indices.numpy()]
    
    # Apply weights to top-k predictions and sum
    weights_expanded = weights.unsqueeze(-1)
    predicted_protein_at_spots = torch.sum(weights_expanded * torch.tensor(topk_protein_preds), dim=1)
    print(f"Predicted proteins at spots shape: {predicted_protein_at_spots.shape}")
    
    return predicted_protein_at_spots, scRNA_pred


def normalize_data(x, low=1e-8, high=1):
    """
    Normalize data to a specified range.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Data to normalize
    low : float, default=1e-8
        Lower bound of the normalized range
    high : float, default=1
        Upper bound of the normalized range
        
    Returns:
    --------
    numpy.ndarray
        Normalized data
    """
    MIN = np.min(x)
    MAX = np.max(x)
    x = low + (x - MIN) / (MAX - MIN) * (high - low)
    return x



def prepare_validation_data(scRNA_pred, scP_truth, predicted_protein_at_spots, fake_p_file, 
                            pred_prefix='pred_protein_'):
    """
    Prepare validation data by aligning predicted and measured proteins.
    
    Parameters:
    -----------
    scRNA_pred : AnnData
        scRNA data with predicted proteins
    scP_truth : AnnData
        Spatial proteomics ground truth data
    predicted_protein_at_spots : torch.Tensor
        Predicted protein expression at spots
    fake_p_file : str
        Path to fake protein file
    pred_prefix : str
        Prefix for prediction columns (default 'pred_protein_', can be 'rand_pred_protein_')
        
    Returns:
    --------
    tuple
        (scP_truth with added columns, overlap_MYIDs list, pred_MYID_list)
    """
    # Normalize data
    scP_truth.X = normalize_data(scP_truth.X)
    # convert sparse matrix to dense if needed for scRNA_pred
    if hasattr(scRNA_pred.X, 'toarray'):
        scRNA_pred.X = scRNA_pred.X.toarray()
    
    scRNA_pred.X = normalize_data(scRNA_pred.X)
    fake_p = sc.read_h5ad(fake_p_file)
    scRNA_pred.uns["protein_predicted_MYID"] = fake_p.var['MYID']
    
    # Get protein IDs
    pred_MYID_list = [int(myid) for myid in scRNA_pred.uns['protein_predicted_MYID']]
    truth_MYID_list = list(scP_truth.var["MYID"].astype(int))
    overlap_MYIDs = list(set(pred_MYID_list).intersection(truth_MYID_list))
    
    print(f"Number of overlapping proteins: {len(overlap_MYIDs)}")
    print(f"Using prediction prefix: {pred_prefix}")
    
    # Convert predictions to DataFrame
    df_pred = pd.DataFrame(predicted_protein_at_spots.numpy(), columns=pred_MYID_list)
    
    # Add predicted protein columns to ground truth object in bulk
    pred_columns = {f"{pred_prefix}{myid}": df_pred[myid].values for myid in pred_MYID_list}
    scP_truth.obs = pd.concat([scP_truth.obs, pd.DataFrame(pred_columns, index=scP_truth.obs.index)], axis=1)
    
    # Add measured protein columns to ground truth object (only once if not already present)
    if f"measured_protein_{overlap_MYIDs[0]}" not in scP_truth.obs.columns:
        measured_columns = {}
        for myid in overlap_MYIDs:
            measured_col_name = f"measured_protein_{myid}"
            protein_idx = np.where(scP_truth.var["MYID"].astype(int) == myid)[0][0]
            measured_columns[measured_col_name] = scP_truth.X[:, protein_idx].flatten()
        
        scP_truth.obs = pd.concat([scP_truth.obs, pd.DataFrame(measured_columns, index=scP_truth.obs.index)], axis=1)
    
    return scP_truth, overlap_MYIDs, pred_MYID_list


def calculate_mmd_scores(truth_np, pred_prot_np, blur_values=None):
    """
    Calculate Maximum Mean Discrepancy (MMD) between truth and predicted distributions.
    
    Parameters:
    -----------
    truth_np : numpy.ndarray
        Ground truth protein expression values
    pred_prot_np : numpy.ndarray
        Predicted protein expression values
    blur_values : list, default=None
        List of blur values to test. If None, uses [0.01, 0.05, 0.1, 0.5]
        
    Returns:
    --------
    dict
        Dictionary of blur values to MMD scores
    """
    try:
        from geomloss import SamplesLoss
    except ImportError:
        print("geomloss not installed. Install with: pip install geomloss")
        return None
        
    if blur_values is None:
        blur_values = [0.01, 0.05, 0.1, 0.5]
    
    truth_tensor = torch.tensor(truth_np).float()
    pred_tensor = torch.tensor(pred_prot_np).float()
    
    mmd_scores = {}
    for blur in blur_values:
        loss = SamplesLoss("gaussian", blur=blur)  # or "energy", "sinkhorn"
        mmd_value = loss(truth_tensor, pred_tensor)
        mmd_scores[blur] = mmd_value.item()
    
    return mmd_scores



def get_protein_name(myid, adata):
    try:
        return adata.var.loc[adata.var["MYID"] == myid, "Protein_y"].values[0]
    except:
        try:
            return adata.var.loc[adata.var["MYID"] == myid, "Protein"].values[0]
        except:
                return f"MYID_{myid}"   


def compute_correlations_with_prefix(obs_df, myid_list, label, pred_prefix='pred_protein_'):
    """
    Compute correlations between predicted and measured proteins with custom prefix.
    
    Parameters:
    -----------
    obs_df : pd.DataFrame
        Observations dataframe with protein columns
    myid_list : list
        List of protein MYIDs to compute correlations for
    label : str
        Label for this analysis
    pred_prefix : str
        Prefix for prediction columns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with correlation results
    """
    from scipy.stats import pearsonr, spearmanr
    
    correlations = []
    
    for myid in myid_list:
        pred_col = f"{pred_prefix}{myid}"
        measured_col = f"measured_protein_{myid}"
        
        if pred_col not in obs_df.columns or measured_col not in obs_df.columns:
            continue
        
        # Get values and remove NaNs
        pred_vals = obs_df[pred_col].values
        meas_vals = obs_df[measured_col].values
        
        mask = ~(np.isnan(pred_vals) | np.isnan(meas_vals))
        
        if np.sum(mask) < 10:
            continue
        
        pred_clean = pred_vals[mask]
        meas_clean = meas_vals[mask]
        
        # Calculate correlations
        pearson_corr, _ = pearsonr(pred_clean, meas_clean)
        spearman_corr, _ = spearmanr(pred_clean, meas_clean)
        
        correlations.append({
            'MYID': myid,
            'Pearson': pearson_corr,
            'Spearman': spearman_corr
        })
    
    return pd.DataFrame(correlations)


    



def plot_single_protein_spatial(adata, img_path, myid, 
                               plot_type='measured',  # 'measured' or 'predicted'
                               region=None, 
                               cell_type_col=None, 
                               cell_type_val=None,
                               scale=1.0, spot_radius=6, 
                               fig_size=(8, 6), dpi=100,
                               padding=50,
                               use_bins=False,
                               n_bins=4,
                               cmap='hot',
                               alpha=0.8,
                               vmin=None,
                               vmax=None,
                               background_alpha=1.0):  # Added parameter
    """
    Plot single protein expression (measured OR predicted) overlaid on H&E image.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.patches import Circle
    import numpy as np
    import pandas as pd
    
    # Load the H&E image
    if img_path is not None:  # Fixed the condition
        img = plt.imread(img_path)
    
    # Get protein name
    try:
        protein_name = adata.var.loc[adata.var["MYID"] == myid, "Protein_y"].values[0]
    except:
        try:
            protein_name = adata.var.loc[adata.var["MYID"] == myid, "Protein"].values[0]
        except:
            protein_name = f"MYID_{myid}"
    
    # Set column names
    if plot_type == 'measured':
        col_name = f"measured_protein_{myid}"
        title_prefix = "Measured"
    elif plot_type == 'predicted':
        col_name = f"pred_protein_{myid}"
        title_prefix = "Predicted"
    else:
        raise ValueError("plot_type must be either 'measured' or 'predicted'")
    
    # Check if column exists
    if col_name not in adata.obs.columns:
        raise ValueError(f"Column {col_name} not found in adata.obs")
    
    # Filter observations - KEEP AS DATAFRAME
    mask = np.ones(len(adata.obs), dtype=bool)
    
    # Filter by region if specified
    if region is not None:
        region_mask = adata.obs['region'] == region
        mask = mask & region_mask
    
    # Filter by cell type if specified
    if cell_type_col is not None:
        cell_type_mask = adata.obs[cell_type_col] == cell_type_val
        mask = mask & cell_type_mask
        # if cell_type not in ['Tumor_1', 'ImmuneCells_1']:
        #     raise ValueError("cell_type must be either 'Tumor_1' or 'ImmuneCells_1'")
        
      
       
    
    # Apply filter - ENSURE IT REMAINS A DATAFRAME
    filtered_obs = adata.obs[mask].copy()  # Use .copy() to ensure it's a DataFrame
    
    # Skip if no observations pass the filters
    if len(filtered_obs) == 0:
        raise ValueError(f"No observations match the specified filters")
    
    # Calculate bounding box of data points
    x_coords = filtered_obs['x'].values * scale
    y_coords = filtered_obs['y'].values * scale
    
    # Calculate min and max coordinates with padding
    x_min = max(0, int(x_coords.min()) - padding)
    y_min = max(0, int(y_coords.min()) - padding)
    x_max = min(img.shape[1], int(x_coords.max()) + padding)
    y_max = min(img.shape[0], int(y_coords.max()) + padding)
    
    # Crop the image
    cropped_img = img[y_min:y_max, x_min:x_max]
    
    # Create figure - single plot
    fig, ax = plt.subplots(1, 1, figsize=fig_size, dpi=dpi)
    
    # Get values
    values = filtered_obs[col_name].values
    
    # Create title text
    region_text = f" Region {region}" if region is not None else ""
    cell_text = f" {cell_type_val}" if cell_type_val is not None else ""
    filter_text = f" ({cell_text})" if region is not None or cell_type_val is not None else ""
    bin_text = " (Binned)" if use_bins else ""
    
    if use_bins:
        # Discretize into bins
        try:
            value_bins = pd.qcut(values, q=n_bins, labels=False)
        except ValueError:
            value_bins = pd.cut(values, bins=n_bins, labels=False)
        
        norm = mcolors.Normalize(vmin=0, vmax=n_bins-1)
        colormap = plt.cm.get_cmap(cmap)  # Fixed deprecation warning
        
        # Plot background image
        ax.imshow(cropped_img, alpha=background_alpha)  # Added background_alpha
        ax.set_title(f"{title_prefix} {protein_name}{filter_text}{bin_text}", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Plot spots - USE iloc FOR PROPER INDEXING
        for i, (idx, row) in enumerate(filtered_obs.iterrows()):
            x = (row['x'] * scale) - x_min
            y = (row['y'] * scale) - y_min
            bin_val = value_bins[i]  # Use array index instead of .loc
            
            if 0 <= x < cropped_img.shape[1] and 0 <= y < cropped_img.shape[0]:
                circ = Circle((x, y), radius=spot_radius, 
                            color=colormap(norm(bin_val)), 
                            linewidth=0, alpha=alpha)
                ax.add_patch(circ)
        
        # Add colorbar
        sm = cm.ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_ticks(np.linspace(0, n_bins-1, n_bins))
        cbar.set_ticklabels([f"Bin {i+1}" for i in range(n_bins)])
        
    else:
        # Continuous values
        if vmin is None:
            vmin = np.percentile(values, 2)
        if vmax is None:
            vmax = np.percentile(values, 98)
        
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        colormap = plt.cm.get_cmap(cmap)  # Fixed deprecation warning
        
        # Plot background image
        ax.imshow(cropped_img, alpha=background_alpha)  # Added background_alpha
        ax.set_title(f"{title_prefix} {protein_name}{filter_text}", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Plot spots - USE ENUMERATION FOR PROPER INDEXING
        for i, (idx, row) in enumerate(filtered_obs.iterrows()):
            x = (row['x'] * scale) - x_min
            y = (row['y'] * scale) - y_min
            value = values[i]  # Use array index
            
            if 0 <= x < cropped_img.shape[1] and 0 <= y < cropped_img.shape[0]:
                circ = Circle((x, y), radius=spot_radius, 
                            color=colormap(norm(value)), 
                            linewidth=0, alpha=alpha)
                ax.add_patch(circ)
        
        # Add colorbar
        sm = cm.ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label(f'{protein_name} Expression', fontsize=14)
    
    plt.tight_layout()
    
    return fig, ax, (x_min, y_min, x_max, y_max), values


def plot_top_proteins_comparison(adata, img_path, top_myids, 
                               region=None, cell_type_col=None,cell_type_val=None,
                               scale=1.0, spot_radius=4,
                               padding=50, cmap='hot', alpha=0.8,
                               use_bins=False, n_bins=4,
                               main_fig_size=(25, 10), dpi=100,
                               background_alpha=1.0):  # Added parameter
    """
    Plot top 5 proteins in two rows: measured (top) and predicted (bottom).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Circle
    import matplotlib.colors as mcolors
    from scipy.stats import pearsonr
    
    # Create figure with subplots: 2 rows, 5 columns
    fig, axes = plt.subplots(2, 5, figsize=main_fig_size, dpi=dpi)
    
    # Load image once
    img = plt.imread(img_path)
    
    # Calculate shared color scale across all proteins if not using bins
    if not use_bins:
        all_measured_values = []
        all_predicted_values = []
        
        for myid in top_myids:
            meas_col = f"measured_protein_{myid}"
            pred_col = f"pred_protein_{myid}"
            
            if meas_col in adata.obs.columns and pred_col in adata.obs.columns:
                # Apply same filtering as in single function
                mask = np.ones(len(adata.obs), dtype=bool)
                if region is not None:
                    mask = mask & (adata.obs['region'] == region)
                if cell_type_col is not None:
                    mask = mask & (adata.obs[cell_type_col] == cell_type_val)
                
                filtered_obs = adata.obs[mask]
                if len(filtered_obs) > 0:
                    all_measured_values.extend(filtered_obs[meas_col].values)
                    all_predicted_values.extend(filtered_obs[pred_col].values)
        
        # Calculate shared vmin and vmax
        all_values = all_measured_values + all_predicted_values
        shared_vmin = np.percentile(all_values, 2)
        shared_vmax = np.percentile(all_values, 98)
    else:
        shared_vmin = None
        shared_vmax = None
    
    # Store correlation values for each protein
    correlation_values = {}
    

    # Helper function to plot on subplot
    def plot_on_subplot(ax, myid, plot_type, row_idx, col_idx):
        try:
            # Get protein name
            try:
                protein_name = adata.var.loc[adata.var["MYID"] == myid, "Protein_y"].values[0]
            except:
                try:
                    protein_name = adata.var.loc[adata.var["MYID"] == myid, "Protein"].values[0]
                except:
                    protein_name = f"MYID_{myid}"
            
            # Get column name
            if plot_type == 'measured':
                col_name = f"measured_protein_{myid}"
                title_prefix = "Measured"
            else:
                col_name = f"pred_protein_{myid}"
                title_prefix = "Predicted"
            
            # Filter data
            mask = np.ones(len(adata.obs), dtype=bool)
            if region is not None:
                mask = mask & (adata.obs['region'] == region)
            if cell_type_col is not None:
                mask = mask & (adata.obs[cell_type_col] == cell_type_val)
            
            filtered_obs = adata.obs[mask].copy()
            
            if len(filtered_obs) == 0:
                ax.text(0.5, 0.5, f"No Data\n{protein_name}", 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # Calculate correlation for this protein (only once per protein)
            if plot_type == 'measured' and myid not in correlation_values:
                meas_col = f"measured_protein_{myid}"
                pred_col = f"pred_protein_{myid}"
                
                if meas_col in adata.obs.columns and pred_col in adata.obs.columns:
                    meas_vals = filtered_obs[meas_col].values
                    pred_vals = filtered_obs[pred_col].values
                    
                    # Remove NaN values
                    valid_mask = ~(np.isnan(meas_vals) | np.isnan(pred_vals))
                    if np.sum(valid_mask) > 1:
                        corr, _ = pearsonr(meas_vals[valid_mask], pred_vals[valid_mask])
                        correlation_values[myid] = corr
                    else:
                        correlation_values[myid] = np.nan
                else:
                    correlation_values[myid] = np.nan
            
            # Calculate crop coordinates
            x_coords = filtered_obs['x'].values * scale
            y_coords = filtered_obs['y'].values * scale
            
            x_min = max(0, int(x_coords.min()) - padding)
            y_min = max(0, int(y_coords.min()) - padding)
            x_max = min(img.shape[1], int(x_coords.max()) + padding)
            y_max = min(img.shape[0], int(y_coords.max()) + padding)
            
            # Crop image
            cropped_img = img[y_min:y_max, x_min:x_max]
            
            # Get values
            values = filtered_obs[col_name].values
            
            # Setup colormap
            if use_bins:
                try:
                    value_bins = pd.qcut(values, q=n_bins, labels=False)
                except ValueError:
                    value_bins = pd.cut(values, bins=n_bins, labels=False)
                norm = mcolors.Normalize(vmin=0, vmax=n_bins-1)
            else:
                norm = mcolors.Normalize(vmin=shared_vmin, vmax=shared_vmax)
            
            colormap = plt.cm.get_cmap(cmap)
            
            # Plot background image with optional transparency
            ax.imshow(cropped_img, alpha=background_alpha)
            
            # Add correlation to title if it's the predicted row
            if plot_type == 'predicted' and myid in correlation_values:
                corr_val = correlation_values[myid]
                if not np.isnan(corr_val):
                    title_text = f"{title_prefix}\n{protein_name}\nPCC: {corr_val:.3f}"
                else:
                    title_text = f"{title_prefix}\n{protein_name}\nPCC: N/A"
            else:
                title_text = f"{title_prefix}\n{protein_name}"
            
            ax.set_title(title_text, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Add spots
            for i, (idx, row) in enumerate(filtered_obs.iterrows()):
                x = (row['x'] * scale) - x_min
                y = (row['y'] * scale) - y_min
                
                if use_bins:
                    color_val = value_bins[i]
                else:
                    color_val = values[i]
                
                if 0 <= x < cropped_img.shape[1] and 0 <= y < cropped_img.shape[0]:
                    circ = Circle((x, y), radius=spot_radius, 
                                color=colormap(norm(color_val)), 
                                linewidth=0, alpha=alpha)
                    ax.add_patch(circ)
            
        except Exception as e:
            print(f"Error plotting {plot_type} protein {myid}: {e}")
            ax.text(0.5, 0.5, f"Error\nMYID {myid}", 
                   ha='center', va='center', transform=ax.transAxes)
    
    # Plot measured proteins (top row)
    for i, myid in enumerate(top_myids):
        plot_on_subplot(axes[0, i], myid, 'measured', 0, i)
    
    # Plot predicted proteins (bottom row)
    for i, myid in enumerate(top_myids):
        plot_on_subplot(axes[1, i], myid, 'predicted', 1, i)
    
    # Add single colorbar on the right side of the entire figure
    if use_bins:
        norm = mcolors.Normalize(vmin=0, vmax=n_bins-1)
        colormap = plt.cm.get_cmap(cmap)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        
        # Create colorbar axis
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.2])  # [left, bottom, width, height]
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_ticks(np.linspace(0, n_bins-1, n_bins))
        cbar.set_ticklabels([f"Bin {i+1}" for i in range(n_bins)])
        cbar.set_label('Expression Level', fontsize=12)
    else:
        norm = mcolors.Normalize(vmin=shared_vmin, vmax=shared_vmax)
        colormap = plt.cm.get_cmap(cmap)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        
        # Create colorbar axis
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.3])  # [left, bottom, width, height]
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Expression Level', fontsize=12)
    
    # Add overall title
    region_text = f" Region {region}" if region is not None else ""
    cell_text = f" {cell_type_val}" if cell_type_val is not None else ""
    filter_text = f" ({cell_text})" if region is not None or cell_type_val is not None else ""
    
    # plt.suptitle(f"Top 5 Proteins: Measured vs Predicted", 
    #              fontsize=16, y=0.98)
    
    # Adjust layout to make room for colorbar
    plt.subplots_adjust(right=0.9, wspace=0.05)
    
    return fig, axes, correlation_values



def initialize_random_embeddings(rna_shape, protein_shape, seed=42):
    """Initialize random embeddings with the same shape as the original embeddings."""
    np.random.seed(seed)
    rna_embeddings = np.random.randn(*rna_shape).astype(np.float32)
    protein_embeddings = np.random.randn(*protein_shape).astype(np.float32)
    
    print(f"Random RNA embeddings shape: {rna_embeddings.shape}")
    print(f"Random Protein embeddings shape: {protein_embeddings.shape}")
    
    return rna_embeddings, protein_embeddings

def run_analysis(rna_embeddings, protein_embeddings,  scRNA_pred_path, k, fake_p_file_path, scP_truth_path, 
                 label="", region=None, cell_type_col=None, cell_type_val=None, 
                 pred_prefix='pred_protein_'):
    """
    Run the complete analysis pipeline and return results.
    
    Parameters:
    -----------
    rna_embeddings : np.ndarray
        RNA embeddings
    protein_embeddings : np.ndarray
        Protein embeddings
    label : str
        Label for this analysis run
    region : int, optional
        Region to filter
    cell_type_col : str, optional
        Column name for cell type filtering
    cell_type_val : str, optional
        Cell type value to filter
    pred_prefix : str
        Prefix for prediction columns (default 'pred_protein_', can be 'rand_pred_protein_')
        
    Returns:
    --------
    tuple
        (results dict, scP_truth AnnData)
    """
    print(f"\n{'='*60}")
    print(f"Running analysis: {label}")
    print(f"Using prediction prefix: {pred_prefix}")
    print(f"{'='*60}")
    
    # Compute spot-level predictions
    print("Computing spot-level predictions...")
    predicted_protein_at_spots, scRNA_pred = compute_spot_level_predictions(
        rna_embeddings, protein_embeddings, scRNA_pred_path, k=k, normalize=False
    )

    # Load ground truth data
    print("Loading ground truth data...")
    scP_truth = sc.read_h5ad(scP_truth_path)

    # Prepare validation data with specified prefix
    print("Preparing validation data...")
    scP_truth, overlap_MYIDs, pred_MYID_list = prepare_validation_data(
        scRNA_pred, scP_truth, predicted_protein_at_spots, fake_p_file_path, pred_prefix=pred_prefix
    )

    # Compute correlations (using the specified prefix)
    print("Computing correlations...")
    df_overall = compute_correlations_with_prefix(scP_truth.obs, overlap_MYIDs, label, pred_prefix=pred_prefix)

    if region is not None:
        if cell_type_col is not None and cell_type_val is not None:
            print(f"\nFiltering for region {region} and cell type {cell_type_val} in column {cell_type_col}...")
            data_subset = scP_truth.obs[
                (scP_truth.obs['region'] == region) & 
                (scP_truth.obs[cell_type_col] == cell_type_val)
            ]
            df_overall = compute_correlations_with_prefix(data_subset, overlap_MYIDs, label, pred_prefix=pred_prefix)
            print(f"\nCorrelation results for region {region} and cell type {cell_type_val}:")
    
    df_overall.set_index('MYID', inplace=True)
    top5_proteins = df_overall.nlargest(5, 'Pearson')
    
    results = {
        'df_overall': df_overall,
        'top5_proteins': top5_proteins,
        'mean_pearson': df_overall["Pearson"].mean(),
        'mean_spearman': df_overall["Spearman"].mean(),
        'top5_mean_pearson': top5_proteins["Pearson"].mean(),
        'scP_truth': scP_truth,
        'overlap_MYIDs': overlap_MYIDs,
        'pred_MYID_list': pred_MYID_list
    }
    
    return results, scP_truth


def get_protein_name(myid, prot_id_to_name):
    """Get protein name from MYID using the mapping dictionary."""
    return prot_id_to_name.get(myid, f"Protein_{myid}")


def compute_compartment_specific_morans(scP_truth, protein_ids, prot_id_to_name,
                                        compartment_col='sc/sn compartments',
                                        pred_prefix='pred_protein_'):
    """
    Calculate Moran's I for each protein WITHIN each compartment.
    
    Parameters:
    -----------
    pred_prefix : str
        Prefix for prediction columns (default 'pred_protein_', can be 'rand_pred_protein_')
    """
    # Remove NaN compartments
    scP_clean = scP_truth[~scP_truth.obs[compartment_col].isna()].copy()
    compartments = scP_clean.obs[compartment_col].unique()
    
    print(f"Computing compartment-specific Moran's I for {pred_prefix}...")
    print(f"Compartments: {compartments}")
    
    comp_moran_results = {comp: {} for comp in compartments}
    
    for comp in compartments:
        print(f"\n  Processing {comp}...")
        
        # Subset to this compartment
        comp_data = scP_clean[scP_clean.obs[compartment_col] == comp].copy()
        
        if len(comp_data) < 50:
            print(f"    Skipping {comp}: too few cells ({len(comp_data)})")
            continue
        
        # Build spatial graph for this compartment
        if "spatial_connectivities" not in comp_data.obsp:
            sq.gr.spatial_neighbors(comp_data, coord_type="generic", spatial_key="spatial")
        
        # Get predicted protein columns
        pred_cols = [f"{pred_prefix}{myid}" for myid in protein_ids 
                    if f"{pred_prefix}{myid}" in comp_data.obs.columns]
        
        # Calculate Moran's I
        sq.gr.spatial_autocorr(
            comp_data,
            mode="moran",
            genes=pred_cols,
            layer=None,
            attr="obs",
        )
        
        # Store results
        morans = pd.DataFrame({
            "Moran_I": comp_data.uns["moranI"]["I"],
            "p_value": comp_data.uns["moranI"]["pval_norm"]
        }, index=pred_cols)
        
        # Convert to protein names
        for pred_col in pred_cols:
            myid = int(pred_col.split('_')[-1])
            protein_name = get_protein_name(myid, prot_id_to_name)
            comp_moran_results[comp][protein_name] = morans.loc[pred_col, 'Moran_I']
        
        print(f"    Computed Moran's I for {len(pred_cols)} proteins")
    
    return comp_moran_results




def compute_compartment_markers_with_comp_spatial(scP_truth, protein_ids, 
                                                   prot_id_to_name,
                                                   compartment_col='sc/sn compartments',
                                                   comp_moran_results=None,
                                                   auc_threshold=0.6):
    """
    Returns BOTH filtered and unfiltered results.
    """
#    Remove NaN compartments first
    scP_clean = scP_truth[~scP_truth.obs[compartment_col].isna()].copy()
    compartments = scP_clean.obs[compartment_col].unique()
    
    print(f"Compartments: {compartments}")
    print(f"Analyzing {len(protein_ids)} proteins...")
    
    marker_results = []
    
    for myid in protein_ids:
        pred_col = f"pred_protein_{myid}"
        
        if pred_col not in scP_clean.obs.columns:
            continue
        
        pred_vals = scP_clean.obs[pred_col].values
        
        if np.std(pred_vals) == 0:
            continue
        
        protein_name = get_protein_name(myid, prot_id_to_name)
        
        # Calculate AUC for each compartment (one-vs-rest)
        for comp in compartments:
            try:
                y_true = (scP_clean.obs[compartment_col] == comp).astype(int)
                
                if y_true.sum() < 10 or (1-y_true).sum() < 10:
                    continue
                
                auc = roc_auc_score(y_true, pred_vals)
                
                # Get compartment-specific Moran's I
                moran_i = np.nan
                if comp_moran_results is not None and comp in comp_moran_results:
                    if protein_name in comp_moran_results[comp]:
                        moran_i = comp_moran_results[comp][protein_name]
                
                marker_results.append({
                    'MYID': myid,
                    'Protein_Name': protein_name,
                    'Compartment': comp,
                    'AUC': auc,
                    'Moran_I_in_compartment': moran_i,
                    'n_positive': y_true.sum(),
                    'n_negative': (1-y_true).sum()
                })
                
            except Exception as e:
                continue
    
    
    df_markers = pd.DataFrame(marker_results)
    
    if len(df_markers) == 0:
        print("WARNING: No markers found!")
        return df_markers, df_markers  # Return both as empty
    
    # Filter for high AUC markers and valid Moran's I
    df_markers_filtered = df_markers[
        (df_markers['AUC'] > auc_threshold) & 
        (~df_markers['Moran_I_in_compartment'].isna())
    ].copy()
    
    df_markers_filtered = df_markers_filtered.sort_values(
        ['Compartment', 'Moran_I_in_compartment'],
        ascending=[True, False]
    )
    
    print(f"\nFound {len(df_markers_filtered)} markers with AUC > {auc_threshold}")
    print(f"Breakdown by compartment:")
    compartments = df_markers['Compartment'].unique()
    for comp in compartments:
        if pd.isna(comp):
            continue
        n = len(df_markers_filtered[df_markers_filtered['Compartment'] == comp])
        if n > 0:
            mean_moran = df_markers_filtered[df_markers_filtered['Compartment'] == comp]['Moran_I_in_compartment'].mean()
            print(f"  {comp}: {n} markers (mean Moran's I: {mean_moran:.3f})")
    
    # Return both unfiltered and filtered
    return df_markers, df_markers_filtered

def compute_compartment_markers_with_random_embeddings(scP_truth, protein_ids, 
                                                        prot_id_to_name,
                                                        compartment_col='sc/sn compartments',
                                                        comp_moran_results=None,
                                                        comp_moran_random_results=None,
                                                        auc_threshold=0.6):
    """
    Compute compartment markers comparing predicted vs random embedding predictions.
    
    Parameters:
    -----------
    scP_truth : AnnData
        Should contain both 'pred_protein_*' and 'rand_pred_protein_*' columns
    comp_moran_results : dict
        Moran's I results for predicted proteins
    comp_moran_random_results : dict
        Moran's I results for random embedding proteins
    """
    # Remove NaN compartments first
    scP_clean = scP_truth[~scP_truth.obs[compartment_col].isna()].copy()
    compartments = scP_clean.obs[compartment_col].unique()
    
    print(f"Compartments: {compartments}")
    print(f"Analyzing {len(protein_ids)} proteins...")
    
    marker_results = []
    
    for myid in protein_ids:
        pred_col = f"pred_protein_{myid}"
        rand_pred_col = f"rand_pred_protein_{myid}"
        
        # Check if both columns exist
        if pred_col not in scP_clean.obs.columns or rand_pred_col not in scP_clean.obs.columns:
            continue
        
        pred_vals = scP_clean.obs[pred_col].values
        rand_pred_vals = scP_clean.obs[rand_pred_col].values
        
        if np.std(pred_vals) == 0 or np.std(rand_pred_vals) == 0:
            continue
        
        protein_name = get_protein_name(myid, prot_id_to_name)
        
        # Calculate AUC for each compartment (one-vs-rest)
        for comp in compartments:
            try:
                y_true = (scP_clean.obs[compartment_col] == comp).astype(int)
                
                if y_true.sum() < 10 or (1-y_true).sum() < 10:
                    continue
                
                # Compute predicted AUC (from learned embeddings)
                auc = roc_auc_score(y_true, pred_vals)
                
                # Compute random AUC (from random embeddings)
                rand_auc = roc_auc_score(y_true, rand_pred_vals)
                
                # Get compartment-specific Moran's I for predicted
                moran_i = np.nan
                if comp_moran_results is not None and comp in comp_moran_results:
                    if protein_name in comp_moran_results[comp]:
                        moran_i = comp_moran_results[comp][protein_name]
                
                # Get compartment-specific Moran's I for random
                moran_i_random = np.nan
                if comp_moran_random_results is not None and comp in comp_moran_random_results:
                    if protein_name in comp_moran_random_results[comp]:
                        moran_i_random = comp_moran_random_results[comp][protein_name]
                
                marker_results.append({
                    'MYID': myid,
                    'Protein_Name': protein_name,
                    'Compartment': comp,
                    'AUC': auc,
                    'Random_AUC': rand_auc,
                    'AUC_improvement': auc - rand_auc,
                    'Moran_I_in_compartment': moran_i,
                    'Moran_I_random': moran_i_random,
                    'n_positive': y_true.sum(),
                    'n_negative': (1-y_true).sum()
                })
                
            except Exception as e:
                print(f"Error processing {protein_name} in {comp}: {e}")
                continue
    
    df_markers = pd.DataFrame(marker_results)
    
    if len(df_markers) == 0:
        print("WARNING: No markers found!")
        return df_markers, df_markers
    
    # Filter for high AUC markers and valid Moran's I
    df_markers_filtered = df_markers[
        (df_markers['AUC'] > auc_threshold) & 
        (~df_markers['Moran_I_in_compartment'].isna())
    ].copy()
    
    df_markers_filtered = df_markers_filtered.sort_values(
        ['Compartment', 'Moran_I_in_compartment'],
        ascending=[True, False] 
    )
    
    print(f"\nFound {len(df_markers_filtered)} markers with AUC > {auc_threshold}")
    print(f"Breakdown by compartment:")
    for comp in compartments:
        if pd.isna(comp):
            continue
        n = len(df_markers_filtered[df_markers_filtered['Compartment'] == comp])
        if n > 0:
            comp_data = df_markers_filtered[df_markers_filtered['Compartment'] == comp]
            mean_moran = comp_data['Moran_I_in_compartment'].mean()
            mean_moran_rand = comp_data['Moran_I_random'].mean()
            mean_auc = comp_data['AUC'].mean()
            mean_rand_auc = comp_data['Random_AUC'].mean()
            mean_improvement = comp_data['AUC_improvement'].mean()
            print(f"  {comp}: {n} markers")
            print(f"    Mean AUC (predicted): {mean_auc:.3f}")
            print(f"    Mean AUC (random embeddings): {mean_rand_auc:.3f}")
            print(f"    Mean AUC Improvement: {mean_improvement:.3f}")
            print(f"    Mean Moran's I (predicted): {mean_moran:.3f}")
            print(f"    Mean Moran's I (random): {mean_moran_rand:.3f}")
    
    return df_markers, df_markers_filtered


def plot_top_spatial_markers_by_compartment(adata, img_path, df_markers_filtered,
                                           compartment_type,
                                           region=3,
                                           scale=1.0, spot_radius=4,
                                           padding=50, cmap='hot', alpha=0.8,
                                           n_bins=4,
                                           main_fig_size=(25, 10), dpi=100,
                                           background_alpha=0.5):
    """
    Plot top 10 spatial markers for a specific compartment using binned values.
    Shows both predicted AUC and random baseline AUC.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Circle
    import matplotlib.colors as mcolors
    import pandas as pd
    
    # Get top 10 markers for this compartment
    comp_markers = df_markers_filtered[df_markers_filtered['Compartment'] == compartment_type].head(10)
    
    if len(comp_markers) == 0:
        print(f"No markers found for {compartment_type}")
        return None, None
    
    protein_names = comp_markers['Protein_Name'].tolist()
    myids = comp_markers['MYID'].tolist()
    aucs = comp_markers['AUC'].tolist()
    rand_aucs = comp_markers['Random_AUC'].tolist()  # Get random AUC values
    
    # Create figure with subplots: 2 rows, 5 columns for up to 10 proteins
    fig, axes = plt.subplots(2, 5, figsize=main_fig_size, dpi=dpi)
    
    # Load image once
    img = plt.imread(img_path)
    
    # Helper function to plot on subplot
    def plot_on_subplot(ax, protein_name, myid, auc, rand_auc, row_idx, col_idx):
        try:
            # Get column name
            col_name = f"pred_protein_{myid}"
            
            # Filter data by compartment AND region
            mask = (adata.obs['sc/sn compartments'] == compartment_type)
            
            # Add region filter if region column exists
            if 'region' in adata.obs.columns and region is not None:
                mask = mask & (adata.obs['region'] == region)
            
            filtered_obs = adata.obs[mask].copy()
            
            if len(filtered_obs) == 0 or col_name not in adata.obs.columns:
                ax.text(0.5, 0.5, f"No Data\n{protein_name}", 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # Get values and create bins
            values = filtered_obs[col_name].values
            
            # Discretize into bins
            try:
                value_bins = pd.qcut(values, q=n_bins, labels=False, duplicates='drop')
            except ValueError:
                value_bins = pd.cut(values, bins=n_bins, labels=False)
            
            # Calculate crop coordinates
            x_coords = filtered_obs['x'].values * scale
            y_coords = filtered_obs['y'].values * scale
            
            x_min = max(0, int(x_coords.min()) - padding)
            y_min = max(0, int(y_coords.min()) - padding)
            x_max = min(img.shape[1], int(x_coords.max()) + padding)
            y_max = min(img.shape[0], int(y_coords.max()) + padding)
            
            # Crop image
            cropped_img = img[y_min:y_max, x_min:x_max]
            
            # Setup colormap for bins
            norm = mcolors.Normalize(vmin=0, vmax=n_bins-1)
            colormap = plt.cm.get_cmap(cmap)
            
            # Plot background image
            ax.imshow(cropped_img, alpha=background_alpha)
            
            # Set title with protein name, predicted AUC, and random baseline AUC
            title_text = f"{protein_name}\nAUC(pred): {auc:.3f}\nAUC(base): {rand_auc:.3f}"
            ax.set_title(title_text, fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Add spots with binned colors
            for i, (idx, row) in enumerate(filtered_obs.iterrows()):
                x = (row['x'] * scale) - x_min
                y = (row['y'] * scale) - y_min
                bin_val = value_bins[i]
                
                if 0 <= x < cropped_img.shape[1] and 0 <= y < cropped_img.shape[0]:
                    circ = Circle((x, y), radius=spot_radius, 
                                color=colormap(norm(bin_val)), 
                                linewidth=0, alpha=alpha)
                    ax.add_patch(circ)
            
        except Exception as e:
            print(f"Error plotting protein {protein_name}: {e}")
            ax.text(0.5, 0.5, f"Error\n{protein_name}", 
                   ha='center', va='center', transform=ax.transAxes)
    
    # Plot proteins in 2x5 grid
    for i, (protein_name, myid, auc, rand_auc) in enumerate(zip(protein_names, myids, aucs, rand_aucs)):
        row = i // 5
        col = i % 5
        plot_on_subplot(axes[row, col], protein_name, myid, auc, rand_auc, row, col)
    
    # Hide any unused subplots
    for i in range(len(protein_names), 10):
        row = i // 5
        col = i % 5
        axes[row, col].set_visible(False)
    
    # Add single colorbar for bins
    norm = mcolors.Normalize(vmin=0, vmax=n_bins-1)
    colormap = plt.cm.get_cmap(cmap)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    sm.set_array([])
    
    # Create colorbar axis
    cbar_ax = fig.add_axes([0.92, 0.15, 0.008, 0.7])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks(np.linspace(0, n_bins-1, n_bins))
    cbar.set_ticklabels([f"Bin {i+1}" for i in range(n_bins)])
    cbar.set_label('Expression Level (Low → High)', fontsize=12)
    
    # Add overall title with compartment name
    # plt.suptitle(f"Top 10 Spatially-Coherent Markers: {compartment_type.title()} Compartment", 
    #              fontsize=16, y=0.95, fontweight='bold')
    
    # Adjust layout
    plt.subplots_adjust(right=0.9, wspace=0.05, hspace=0.3)
    
    return fig, axes


def plot_top_proteins_immune_high_vs_low(adata, img_path, top_myids, 
                                         pcc_pred_dict, pcc_rand_dict,
                                         region=None, 
                                         cell_type_col='sc/sn compartments', 
                                         cell_type_value='lymphoid',
                                         percentile_threshold=50,
                                         scale=1.0, spot_radius=8,
                                         padding=50, alpha=0.9,
                                         main_fig_size=(25, 10), dpi=100,
                                         background_alpha=0.3):
    """
    Plot top 5 proteins showing high vs low expression in lymphoid cells.
    Two rows: measured (top, blue) and predicted (bottom, red).
    Shows PCC for predicted vs measured and random vs measured.
    
    Parameters:
    -----------
    adata : AnnData
        Spatial proteomics data
    img_path : str
        Path to H&E image
    top_myids : list
        List of top 5 MYIDs to plot
    pcc_pred_dict : dict
        Dictionary with MYID as key and PCC (predicted vs measured) as value
    pcc_rand_dict : dict
        Dictionary with MYID as key and PCC (random vs measured) as value
    region : int, optional
        Region to filter
    cell_type_col : str
        Column name for cell type
    cell_type_value : str
        Cell type value to filter (e.g., 'lymphoid')
    percentile_threshold : int
        Percentile threshold for high expression (default 50%)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Circle, Patch
    
    # Create figure with subplots: 2 rows, 5 columns
    fig, axes = plt.subplots(2, 5, figsize=main_fig_size, dpi=dpi)
    
    # Load image once
    img = plt.imread(img_path)
    
    # Define colors for measured (blue) and predicted (red)
    colors_measured = {
        'high': 'blue',  # Blue for high expression
        'low': '#95A5A6'    # Gray for low expression
    }
    
    colors_predicted = {
        'high': 'red',  # Red for high expression
        'low': '#95A5A6'    # Gray for low expression
    }
    
    def plot_on_subplot(ax, myid, plot_type, row_idx, col_idx):
        try:
            # Get protein name
            try:
                protein_name = adata.var.loc[adata.var["MYID"] == myid, "Protein_y"].values[0]
            except:
                try:
                    protein_name = adata.var.loc[adata.var["MYID"] == myid, "Protein"].values[0]
                except:
                    protein_name = f"MYID_{myid}"
            
            # Get column name and colors
            if plot_type == 'measured':
                col_name = f"measured_protein_{myid}"
                title_prefix = "Measured"
                colors = colors_measured
            else:
                col_name = f"pred_protein_{myid}"
                title_prefix = "Predicted"
                colors = colors_predicted
            
            # Filter data by region and cell type
            mask = np.ones(len(adata.obs), dtype=bool)
            if region is not None:
                mask = mask & (adata.obs['region'] == region)
            if cell_type_col is not None:
                mask = mask & (adata.obs[cell_type_col] == cell_type_value)
            
            filtered_obs = adata.obs[mask].copy()
            
            if len(filtered_obs) == 0:
                ax.text(0.5, 0.5, f"No Data\n{protein_name}", 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # Calculate threshold for this protein and plot type
            values = filtered_obs[col_name].values
            threshold = np.percentile(values, percentile_threshold)
            
            # Calculate crop coordinates
            x_coords = filtered_obs['x'].values * scale
            y_coords = filtered_obs['y'].values * scale
            
            x_min = max(0, int(x_coords.min()) - padding)
            y_min = max(0, int(y_coords.min()) - padding)
            x_max = min(img.shape[1], int(x_coords.max()) + padding)
            y_max = min(img.shape[0], int(y_coords.max()) + padding)
            
            # Crop image
            cropped_img = img[y_min:y_max, x_min:x_max]
            
            # Plot background image
            ax.imshow(cropped_img, alpha=background_alpha)
            
            # Build title with PCC values
            if plot_type == 'predicted':
                pcc_pred = pcc_pred_dict[myid]
                pcc_rand = pcc_rand_dict[myid]
                
                if not np.isnan(pcc_pred) and not np.isnan(pcc_rand):
                    title_text = (f"{title_prefix}\n{protein_name}\n"
                                f"PCC(Pred): {pcc_pred:.3f}\n"
                                f"PCC(Rand): {pcc_rand:.3f}")
                else:
                    title_text = f"{title_prefix}\n{protein_name}\nPCC: N/A"
            else:
                title_text = f"{title_prefix}\n{protein_name}"
            
            ax.set_title(title_text, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Plot spots
            for i, (idx, row) in enumerate(filtered_obs.iterrows()):
                x = (row['x'] * scale) - x_min
                y = (row['y'] * scale) - y_min
                
                value = values[i]
                
                # Determine color based on threshold
                if value >= threshold:
                    color = colors['high']
                else:
                    color = colors['low']
                
                if 0 <= x < cropped_img.shape[1] and 0 <= y < cropped_img.shape[0]:
                    circ = Circle((x, y), radius=spot_radius, 
                                color=color, linewidth=0, alpha=alpha)
                    ax.add_patch(circ)
            
            # Add legend (only for first column)
            if col_idx == 0:
                legend_elements = [
                    Patch(facecolor=colors['high'], edgecolor='none', 
                         label=f'High (≥{percentile_threshold}%)'),
                    Patch(facecolor=colors['low'], edgecolor='none', 
                         label=f'Low (<{percentile_threshold}%)')
                ]
                ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
            
        except Exception as e:
            print(f"Error plotting {plot_type} protein {myid}: {e}")
            ax.text(0.5, 0.5, f"Error\nMYID {myid}", 
                   ha='center', va='center', transform=ax.transAxes)
    
    # Plot measured proteins (top row)
    for i, myid in enumerate(top_myids):
        plot_on_subplot(axes[0, i], myid, 'measured', 0, i)
    
    # Plot predicted proteins (bottom row)
    for i, myid in enumerate(top_myids):
        plot_on_subplot(axes[1, i], myid, 'predicted', 1, i)
    
    # Add overall title
    cell_text = f" {cell_type_value}" if cell_type_value is not None else ""
    region_text = f" Region {region}" if region is not None else ""
    
    # plt.suptitle(f"Top 5 Proteins: High vs Low Expression in{cell_text}{region_text}", 
    #              fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    return fig, axes




def simple_permutation_test(df_markers_all, df_markers_filtered, n_permutations=1000):
    """
    Test if high-AUC markers have higher Moran's I than ALL proteins.
    
    Parameters:
    -----------
    df_markers_all : ALL proteins with AUC calculated (unfiltered)
    df_markers_filtered : Only high-AUC markers
    """
    
    compartments = df_markers_filtered['Compartment'].unique()
    results = {}
    
    print("\n" + "="*70)
    print("SPATIAL COHERENCE TEST FOR COMPARTMENT MARKERS")
    print("="*70)
    
    for comp in compartments:
        if pd.isna(comp):
            continue
            
        # HIGH-AUC markers in this compartment
        comp_markers = df_markers_filtered[df_markers_filtered['Compartment'] == comp].copy()
        
        # ALL proteins in this compartment (including low-AUC)
        all_comp_proteins = df_markers_all[
            (df_markers_all['Compartment'] == comp) &
            (~df_markers_all['Moran_I_in_compartment'].isna())
        ].copy()
        
        if len(comp_markers) < 10 or len(all_comp_proteins) < 50:
            continue
        
        # Observed: mean Moran's I of HIGH-AUC markers
        observed_mean_moran = comp_markers['Moran_I_in_compartment'].mean()
        
        # Permutation: sample from ALL proteins (including low-AUC)
        n_markers = len(comp_markers)
        permuted_means = []
        
        for _ in range(n_permutations):
            random_sample = all_comp_proteins.sample(n=min(n_markers, len(all_comp_proteins)), 
                                                     replace=False)
            permuted_means.append(random_sample['Moran_I_in_compartment'].mean())
        
        permuted_means = np.array(permuted_means)
        
        # Calculate p-value (one-sided: observed > expected)
        p_value = (np.sum(permuted_means >= observed_mean_moran) + 1) / (n_permutations + 1)
        
        expected_mean = np.mean(permuted_means)
        expected_std = np.std(permuted_means)
        
        results[comp] = {
            'n_markers': len(comp_markers),
            'n_all_proteins': len(all_comp_proteins),
            'observed_mean_moran': observed_mean_moran,
            'expected_mean_moran': expected_mean,
            'expected_std': expected_std,
            'p_value': p_value
        }
        
        print(f"\n{comp}:")
        print(f"  High-AUC markers: {len(comp_markers)}")
        print(f"  All proteins tested: {len(all_comp_proteins)}")
        print(f"  Observed mean Moran's I (high-AUC): {observed_mean_moran:.4f}")
        print(f"  Expected (all proteins): {expected_mean:.4f} ± {expected_std:.4f}")
        print(f"  P-value: {p_value:.4f}{'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")
    
    return results



def plot_permutation_test_result(df_markers_all, df_markers_filtered, perm_results, 
                                 compartment='nan', n_permutations=1000):
    """
    Visualize the permutation test result for a specific compartment.
    
    Parameters:
    -----------
    df_markers_all : DataFrame with all protein-compartment combinations
    df_markers_filtered : DataFrame with high-AUC markers only
    perm_results : Dictionary with permutation test results
    compartment : Which compartment to visualize
    """
    
    if compartment not in perm_results:
        print(f"No results found for compartment '{compartment}'")
        print(f"Available compartments: {list(perm_results.keys())}")
        return
    
    # Get data for this compartment
    comp_markers = df_markers_filtered[df_markers_filtered['Compartment'] == compartment].copy()
    all_comp_proteins = df_markers_all[
        (df_markers_all['Compartment'] == compartment) &
        (~df_markers_all['Moran_I_in_compartment'].isna())
    ].copy()
    
    # Calculate unique protein counts
    n_markers = len(comp_markers)
    observed_mean = comp_markers['Moran_I_in_compartment'].mean()
    
    np.random.seed(42)
    permuted_means = []
    for _ in range(n_permutations):
        random_sample = all_comp_proteins.sample(n=n_markers, replace=False)
        permuted_means.append(random_sample['Moran_I_in_compartment'].mean())
    
    permuted_means = np.array(permuted_means)
    
    # Get test results
    res = perm_results[compartment]
    p_value = res['p_value']
    expected_mean = res['expected_mean_moran']
    expected_std = res['expected_std']
    # z_score = res['z_score']
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # ===== LEFT PLOT: Null distribution with observed value =====
    ax1 = axes[0]
    
    # Plot histogram of null distribution
    counts, bins, patches = ax1.hist(permuted_means, bins=50, alpha=0.7, 
                                     color='lightblue', edgecolor='black', 
                                     density=True, label='Null distribution')
    
    # Overlay normal curve
    from scipy import stats
    x = np.linspace(permuted_means.min(), permuted_means.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, expected_mean, expected_std), 
            'b--', linewidth=2, label=f'Normal fit (μ={expected_mean:.4f}, σ={expected_std:.4f})')
    
    # Mark observed value
    ax1.axvline(observed_mean, color='red', linestyle='-', linewidth=3, 
               label=f'Observed (high-AUC markers)\nMoran\'s I = {observed_mean:.4f}')
    
    # Mark expected value
    ax1.axvline(expected_mean, color='blue', linestyle='--', linewidth=2,
               label=f'Expected (random)\nMoran\'s I = {expected_mean:.4f}')
    
    # Shade p-value region (upper tail)
    threshold = np.percentile(permuted_means, (1 - p_value) * 100)
    x_fill = x[x >= observed_mean]
    y_fill = stats.norm.pdf(x_fill, expected_mean, expected_std)
    ax1.fill_between(x_fill, y_fill, alpha=0.3, color='red', 
                    label=f'P-value region (p={p_value:.4f})')
    
    ax1.set_xlabel("Mean Moran's I", fontsize=13, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax1.set_title(f"Permutation Test: {compartment} Compartment\n(n={n_permutations} permutations)", 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(False)
    
    # # Add text box with statistics
    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    textstr = f'P-value: {p_value:.4f} {sig}\n' + \
              f'High-AUC markers: {res["n_markers"]} proteins\n' + \
              f'All proteins: {res["n_all_proteins"]} proteins'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    # ax1.text(0.98, 0.97, textstr, transform=ax1.transAxes, fontsize=11,
    #         verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # ===== RIGHT PLOT: Comparison of distributions =====
    ax2 = axes[1]
    
    # Get Moran's I values
    high_auc_morans = comp_markers['Moran_I_in_compartment'].values
    all_morans = all_comp_proteins['Moran_I_in_compartment'].values
    
    # Plot distributions
    ax2.hist(all_morans, bins=40, alpha=0.5, color='gray', 
            label=f'All proteins (n={len(all_morans)})', 
            edgecolor='black', density=True)
    ax2.hist(high_auc_morans, bins=30, alpha=0.7, color='red', 
            label=f'High-AUC markers (n={len(high_auc_morans)})', 
            edgecolor='black', density=True)
    
    # Add mean lines
    ax2.axvline(all_morans.mean(), color='gray', linestyle='--', linewidth=2,
               label=f'Mean (all): {all_morans.mean():.4f}')
    ax2.axvline(high_auc_morans.mean(), color='red', linestyle='-', linewidth=2,
               label=f'Mean (high-AUC): {high_auc_morans.mean():.4f}')
    
    ax2.set_xlabel("Moran's I (within compartment)", fontsize=13, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax2.set_title(f"Distribution Comparison: {compartment}\nHigh-AUC Markers vs All Proteins", 
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(False)
    
    # Add Cohen's d effect size
    from scipy.stats import ttest_ind
    cohens_d = (high_auc_morans.mean() - all_morans.mean()) / np.sqrt(
        (high_auc_morans.std()**2 + all_morans.std()**2) / 2
    )
    t_stat, t_pval = ttest_ind(high_auc_morans, all_morans)
    
    textstr2 = f"Cohen's d: {cohens_d:.3f}\nt-test p: {t_pval:.4e}"
    ax2.text(0.02, 0.97, textstr2, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    if "/" in compartment or "\\" in compartment:
        compartment = compartment.replace("/", "_").replace("\\", "_")


    # plt.savefig(f'permutation_test_{compartment}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"PERMUTATION TEST SUMMARY: {compartment} Compartment")
    print(f"{'='*70}")
    print(f"\nNull Hypothesis: High-AUC markers have the SAME spatial coherence as all proteins")
    print(f"Alternative: High-AUC markers have HIGHER spatial coherence\n")
    print(f"High-AUC Markers:")
    print(f"  Count: {len(high_auc_morans)} ({res['n_all_proteins']} unique proteins)")
    print(f"  Mean Moran's I: {high_auc_morans.mean():.4f}")
    print(f"  Median Moran's I: {np.median(high_auc_morans):.4f}")
    print(f"  Std: {high_auc_morans.std():.4f}\n")
    print(f"All Proteins:")
    print(f"  Count: {len(all_morans)} ({res['n_all_proteins']} unique proteins)")
    print(f"  Mean Moran's I: {all_morans.mean():.4f}")
    print(f"  Median Moran's I: {np.median(all_morans):.4f}")
    print(f"  Std: {all_morans.std():.4f}\n")
    print(f"Test Results:")
   
    print(f"  P-value (one-tailed): {p_value:.4f} {sig}")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print(f"\nInterpretation:")
    if p_value < 0.05:
        print(f"  ✓ High-AUC markers are significantly MORE spatially coherent than expected")
    else:
        print(f"  ✗ No significant difference in spatial coherence")